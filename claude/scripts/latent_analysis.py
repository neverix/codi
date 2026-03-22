#!/usr/bin/env python3
"""
GPU script: collect logit lens (per-layer lm_head) + attention entropy per latent.
Saves results to outputs/latent_data.pt.

Based on probe_latent_token.py inference loop.
"""

import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from torch.nn import functional as F
import json

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from safetensors.torch import load_file

import numpy as np

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

LOGIT_LENS_TOPK = 5


def make_eager_attention_mask(attention_mask, dtype):
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    batch_size, seq_len = attention_mask.shape
    converter = AttentionMaskConverter(is_causal=True, sliding_window=None)
    mask_4d = converter.to_4d(attention_mask, seq_len, dtype=dtype, key_value_length=seq_len)
    min_dtype = torch.finfo(dtype).min
    mask_4d = AttentionMaskConverter._unmask_unattended(mask_4d, min_dtype)
    return mask_4d


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])


def compute_attention_entropy(attentions, attention_mask, batch_size, input_len):
    """Compute entropy of attention distribution from last token over non-padding positions.

    Args:
        attentions: tuple of (batch, heads, q_len, kv_len) per layer
        attention_mask: (batch, input_len) 1=real, 0=pad — only for input tokens
        batch_size: int
        input_len: int (original input length)

    Returns:
        entropy: (batch,) — average entropy across layers and heads
        entropy_input_only: (batch,) — entropy computed only over input token positions
    """
    # Stack: (layers, batch, heads, q_len, kv_len)
    attn_stack = torch.stack(list(attentions), dim=0)
    # Get attention from last query position: (layers, batch, heads, kv_len)
    attn_last = attn_stack[:, :, :, -1, :]

    full_kv_len = attn_last.shape[-1]

    # Build padding mask extended to full kv length
    pad_mask = (attention_mask == 0)  # (batch, input_len)
    if pad_mask.shape[-1] < full_kv_len:
        extra = torch.zeros(batch_size, full_kv_len - pad_mask.shape[-1],
                            dtype=torch.bool, device=pad_mask.device)
        pad_mask_full = torch.cat([pad_mask, extra], dim=-1)
    else:
        pad_mask_full = pad_mask[:, :full_kv_len]

    # --- Full entropy (all non-padding positions) ---
    # Mask padding in attention weights, renormalize
    # pad_mask_full: (batch, kv_len) -> (1, batch, 1, kv_len)
    mask_4d = pad_mask_full.unsqueeze(0).unsqueeze(2)
    attn_masked = attn_last.clone()
    attn_masked[mask_4d.expand_as(attn_masked)] = 0.0
    # Renormalize
    attn_sum = attn_masked.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    attn_normed = attn_masked / attn_sum
    # Entropy: H = -sum(a * log(a))
    log_attn = torch.log(attn_normed.clamp(min=1e-12))
    entropy_per_layer_head = -(attn_normed * log_attn).sum(dim=-1)  # (layers, batch, heads)
    entropy = entropy_per_layer_head.mean(dim=(0, 2))  # (batch,)

    # --- Input-only entropy ---
    input_only_mask = torch.ones(batch_size, full_kv_len, dtype=torch.bool, device=attn_last.device)
    input_only_mask[:, :input_len] = ~pad_mask[:, :input_len]  # True = keep
    # zero out non-input and padding positions
    keep_4d = input_only_mask.unsqueeze(0).unsqueeze(2)
    attn_input = attn_last.clone()
    attn_input[~keep_4d.expand_as(attn_input)] = 0.0
    attn_input_sum = attn_input.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    attn_input_normed = attn_input / attn_input_sum
    log_attn_input = torch.log(attn_input_normed.clamp(min=1e-12))
    entropy_input = -(attn_input_normed * log_attn_input).sum(dim=-1)
    entropy_input_only = entropy_input.mean(dim=(0, 2))  # (batch,)

    return entropy.float(), entropy_input_only.float()


def apply_logit_lens(hidden_states, lm_head, topk=LOGIT_LENS_TOPK):
    """Apply lm_head to each layer's hidden state at the last position.

    Args:
        hidden_states: tuple of (batch, seq_len, hidden_dim) — one per layer (embedding + N layers)
        lm_head: the lm_head module
        topk: number of top predictions to keep

    Returns:
        top_ids: (batch, num_layers, topk)
        top_probs: (batch, num_layers, topk)
    """
    all_ids = []
    all_probs = []
    for h in hidden_states:
        # h: (batch, seq_len, hidden_dim) — take last position
        logits = lm_head(h[:, -1:, :])  # (batch, 1, vocab)
        probs = F.softmax(logits[:, 0, :], dim=-1)  # (batch, vocab)
        vals, ids = torch.topk(probs, k=topk, dim=-1)  # (batch, topk)
        all_ids.append(ids)
        all_probs.append(vals)

    return torch.stack(all_ids, dim=1).int(), torch.stack(all_probs, dim=1).float()


def run_analysis(model_args, data_args, training_args):
    # --- Model setup (same as probe_latent_token.py) ---
    task_type = TaskType.CAUSAL_LM
    if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

    lora_config = LoraConfig(
        task_type=task_type, inference_mode=False,
        r=model_args.lora_r, lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1, target_modules=target_modules, init_lora_weights=True,
    )

    model = CODI(model_args, training_args, lora_config)
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left", use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    model = model.to('cuda').to(torch.bfloat16)

    # --- Dataset ---
    dataset = load_dataset(data_args.data_name)
    test_set = dataset['test']

    question = []
    answer = []
    procedures = []
    for example in test_set:
        question.append(f"{example['question'].strip().replace('  ', ' ')}")
        answer.append(float(example['answer'].replace(',', '')))
        procedures.append(example['cot'])

    eval_step = math.ceil(len(question) / data_args.batch_size)
    print(f"Total examples: {len(question)} | batch_size: {data_args.batch_size} | steps: {eval_step}")

    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(question[i*data_args.batch_size:(i+1)*data_args.batch_size],
                              return_tensors="pt", padding="longest")
        else:
            batch = tokenizer(question[i*data_args.batch_size:],
                              return_tensors="pt", padding="longest")

        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        question_data.append(batch.to(device))

    model.eval()
    lm_head = model.codi.lm_head if hasattr(model.codi, 'lm_head') else model.codi.get_base_model().lm_head

    # Get number of layers from config
    if hasattr(model.codi.config, 'n_layer'):
        num_transformer_layers = model.codi.config.n_layer  # GPT-2
    elif hasattr(model.codi.config, 'num_hidden_layers'):
        num_transformer_layers = model.codi.config.num_hidden_layers
    else:
        raise ValueError("Cannot determine number of layers from config")
    num_hidden_states = num_transformer_layers + 1  # embedding + transformer layers
    print(f"Model has {num_transformer_layers} transformer layers ({num_hidden_states} hidden states)")

    inf_latent_iterations = training_args.inf_latent_iterations
    num_latent_positions = 1 + inf_latent_iterations  # z0 (initial) + iterations

    # Accumulators
    all_logit_lens_ids = []    # list of (batch, num_latent_pos, num_layers, topk)
    all_logit_lens_probs = []
    all_entropy = []           # list of (batch, num_latent_pos)
    all_entropy_input = []
    all_correct = []
    all_predictions = []
    all_answers = []
    all_cot_steps = []

    gen_kwargs = {
        "max_new_tokens": 256, "temperature": 0.1,
        "top_k": 40, "top_p": 0.95, "do_sample": True,
    }

    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        input_len = batch["input_ids"].size(1)

        batch_lens_ids = []   # list of (batch, num_layers, topk) per latent pos
        batch_lens_probs = []
        batch_entropy = []
        batch_entropy_input = []

        with torch.no_grad():
            # --- Encode ---
            past_key_values = None
            eager_attn_mask = make_eager_attention_mask(batch["attention_mask"], dtype=torch.bfloat16)
            outputs = model.codi(
                input_ids=batch["input_ids"], use_cache=True,
                output_hidden_states=True, past_key_values=past_key_values,
                attention_mask=eager_attn_mask, output_attentions=True,
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Logit lens for z0
            ids, probs = apply_logit_lens(outputs.hidden_states, lm_head)
            batch_lens_ids.append(ids)
            batch_lens_probs.append(probs)

            # Attention entropy for z0
            ent, ent_input = compute_attention_entropy(
                outputs.attentions, batch["attention_mask"], batch_size, input_len)
            batch_entropy.append(ent)
            batch_entropy_input.append(ent_input)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # --- Latent iterations ---
            for i in range(inf_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd, use_cache=True,
                    output_hidden_states=True, past_key_values=past_key_values,
                    output_attentions=True,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Logit lens
                ids, probs = apply_logit_lens(outputs.hidden_states, lm_head)
                batch_lens_ids.append(ids)
                batch_lens_probs.append(probs)

                # Attention entropy
                ent, ent_input = compute_attention_entropy(
                    outputs.attentions, batch["attention_mask"], batch_size, input_len)
                batch_entropy.append(ent)
                batch_entropy_input.append(ent_input)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            # Stack per-batch results
            # lens_ids: list of num_latent_pos x (batch, num_layers, topk) -> (batch, num_latent_pos, num_layers, topk)
            all_logit_lens_ids.append(torch.stack(batch_lens_ids, dim=1).cpu())
            all_logit_lens_probs.append(torch.stack(batch_lens_probs, dim=1).cpu())
            all_entropy.append(torch.stack(batch_entropy, dim=1).cpu())
            all_entropy_input.append(torch.stack(batch_entropy_input, dim=1).cpu())

            # --- Generate answer (same as probe_latent_token.py) ---
            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device='cuda')).unsqueeze(0)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0)
            eot_emb = eot_emb.expand(batch_size, -1, -1)

            output = eot_emb
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
            pred_tokens = [[] for _ in range(batch_size)]
            for i in range(gen_kwargs["max_new_tokens"]):
                out = model.codi(
                    inputs_embeds=output, output_hidden_states=False,
                    attention_mask=None, use_cache=True,
                    output_attentions=False, past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]

                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).view(-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")
                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False
                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")
                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).view(-1)

                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True
                if finished.all():
                    break
                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            # Process predictions
            for mini_step, pred_token in enumerate(pred_tokens):
                global_idx = step * data_args.batch_size + mini_step
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                pred_ans = extract_answer_number(decoded_pred)
                is_correct = (not math.isinf(pred_ans)) and (int(pred_ans) == int(answer[global_idx]))
                all_correct.append(is_correct)
                all_predictions.append(decoded_pred)
                all_answers.append(answer[global_idx])

            # Parse CoT steps for this batch
            for mini_step in range(batch_size):
                global_idx = step * data_args.batch_size + mini_step
                if global_idx < len(procedures):
                    cot = procedures[global_idx]
                    steps = []
                    for block in re.findall(r"<<(.+?)>>", cot):
                        parts = block.split("=")
                        if len(parts) >= 2:
                            try:
                                steps.append(float(parts[-1]))
                            except ValueError:
                                pass
                    all_cot_steps.append(steps)
                else:
                    all_cot_steps.append([])

        if (step + 1) % 2 == 0 or step == eval_step - 1:
            n_done = min((step + 1) * data_args.batch_size, len(question))
            print(f"  Step {step+1}/{eval_step} ({n_done}/{len(question)} examples)")

    # --- Concatenate and save ---
    logit_lens_ids = torch.cat(all_logit_lens_ids, dim=0)      # (N, 7, 13, 5)
    logit_lens_probs = torch.cat(all_logit_lens_probs, dim=0)  # (N, 7, 13, 5)
    entropy = torch.cat(all_entropy, dim=0)                     # (N, 7)
    entropy_input = torch.cat(all_entropy_input, dim=0)         # (N, 7)
    correct = torch.tensor(all_correct, dtype=torch.bool)       # (N,)
    answers = torch.tensor(all_answers, dtype=torch.float32)    # (N,)

    print(f"\nShapes:")
    print(f"  logit_lens_ids:   {logit_lens_ids.shape}")
    print(f"  logit_lens_probs: {logit_lens_probs.shape}")
    print(f"  entropy:          {entropy.shape}")
    print(f"  entropy_input:    {entropy_input.shape}")
    print(f"  correct:          {correct.shape}")
    print(f"  answers:          {answers.shape}")
    print(f"  predictions:      {len(all_predictions)}")
    print(f"  cot_steps:        {len(all_cot_steps)}")

    accuracy = correct.float().mean().item()
    print(f"\nAccuracy: {100*accuracy:.2f}%")

    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/latent_data.pt"
    torch.save({
        "logit_lens_top5_ids": logit_lens_ids,
        "logit_lens_top5_probs": logit_lens_probs,
        "attention_entropy": entropy,
        "attention_entropy_input_only": entropy_input,
        "cot_steps": all_cot_steps,
        "correct": correct,
        "predictions": all_predictions,
        "answers": answers,
    }, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    run_analysis(model_args, data_args, training_args)
