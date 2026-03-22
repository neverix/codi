#!/usr/bin/env python3
"""
GPU script: run inference with inf_latent_iterations=0..6, record accuracy per condition.
Saves results to outputs/early_stopping.json.

Based on probe_latent_token.py inference loop.
"""

import logging
import math
import re
import os
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from torch.nn import functional as F

from peft import LoraConfig, TaskType
from datasets import load_dataset
from safetensors.torch import load_file

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


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


def compute_accuracy(gold, pred):
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1
    return acc / len(gold)


def run_eval_with_iterations(model, tokenizer, question_data, answer, n_iterations,
                              training_args, gen_kwargs, data_args):
    """Run full eval with a specific number of latent iterations. Returns accuracy."""
    model.eval()
    ans_pred_list = []

    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)

        with torch.no_grad():
            past_key_values = None
            eager_attn_mask = make_eager_attention_mask(batch["attention_mask"], dtype=torch.bfloat16)
            outputs = model.codi(
                input_ids=batch["input_ids"], use_cache=True,
                output_hidden_states=True, past_key_values=past_key_values,
                attention_mask=eager_attn_mask, output_attentions=False,
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # Latent iterations
            for i in range(n_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd, use_cache=True,
                    output_hidden_states=True, past_key_values=past_key_values,
                    output_attentions=False,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            # EOT embedding
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

            for mini_step, pred_token in enumerate(pred_tokens):
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                ans_pred_list.append(extract_answer_number(decoded_pred))

    accuracy = compute_accuracy(answer, ans_pred_list)
    return accuracy


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- Model setup ---
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
    for example in test_set:
        question.append(f"{example['question'].strip().replace('  ', ' ')}")
        answer.append(float(example['answer'].replace(',', '')))

    eval_step = math.ceil(len(question) / data_args.batch_size)

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

    gen_kwargs = {
        "max_new_tokens": 256, "temperature": 0.1,
        "top_k": 40, "top_p": 0.95, "do_sample": True,
    }

    # --- Run for each iteration count ---
    results = {}
    max_iterations = training_args.inf_latent_iterations  # default 6

    for n_iter in range(0, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Running with inf_latent_iterations = {n_iter}")
        print(f"{'='*60}")

        accuracy = run_eval_with_iterations(
            model, tokenizer, question_data, answer, n_iter,
            training_args, gen_kwargs, data_args,
        )
        results[n_iter] = round(100 * accuracy, 2)
        print(f"  iterations={n_iter}: accuracy={100*accuracy:.2f}%")

    # --- Save ---
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/early_stopping.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_path}")
    print(f"Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
