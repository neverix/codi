#!/usr/bin/env python3
"""
GPU script: collect top-K attended INPUT token positions per latent position.
No generation needed — just encoding + latent iterations with attention.
Saves to outputs/attention_tracking.pt.
"""

import math
import re
import os

import torch
import transformers
from torch.nn import functional as F

from peft import LoraConfig, TaskType
from datasets import load_dataset
from safetensors.torch import load_file

from src.model import CODI, ModelArguments, DataArguments, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ATTN_TOPK = 20  # save top 20 attended input positions per latent


def make_eager_attention_mask(attention_mask, dtype):
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    batch_size, seq_len = attention_mask.shape
    converter = AttentionMaskConverter(is_causal=True, sliding_window=None)
    mask_4d = converter.to_4d(attention_mask, seq_len, dtype=dtype, key_value_length=seq_len)
    min_dtype = torch.finfo(dtype).min
    mask_4d = AttentionMaskConverter._unmask_unattended(mask_4d, min_dtype)
    return mask_4d


def run(model_args, data_args, training_args):
    # --- Model setup ---
    if "gpt2" in model_args.model_name_or_path.lower():
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif any(n in model_args.model_name_or_path.lower() for n in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
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
    model.eval()

    # --- Dataset ---
    dataset = load_dataset(data_args.data_name)
    test_set = dataset['test']
    questions, answers, procedures = [], [], []
    for ex in test_set:
        questions.append(ex['question'].strip().replace('  ', ' '))
        answers.append(float(ex['answer'].replace(',', '')))
        procedures.append(ex['cot'])

    N = len(questions)
    eval_step = math.ceil(N / data_args.batch_size)
    print(f"Total: {N} examples, {eval_step} batches")

    # Tokenize
    question_data = []
    for i in range(eval_step):
        start = i * data_args.batch_size
        end = min(start + data_args.batch_size, N)
        batch = tokenizer(questions[start:end], return_tensors="pt", padding="longest")
        if training_args.remove_eos:
            bot = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot)), dim=1)
        question_data.append(batch.to(device))

    inf_iters = training_args.inf_latent_iterations
    num_latent_pos = 1 + inf_iters  # z0 + iterations

    # Accumulators (per-example)
    all_attn_indices = []   # list of (7, K) tensors
    all_attn_values = []    # list of (7, K) tensors
    all_frac_to_input = []  # list of (7,) tensors — fraction of attention to input tokens
    all_input_tokens = []   # list of list of int — unpadded input token IDs
    # Also save per-latent attention to other latent positions
    all_attn_to_latents = []  # list of (7, max_latent) — attention to each latent position

    for step, batch in enumerate(question_data):
        bs = batch["input_ids"].size(0)
        input_len = batch["input_ids"].size(1)
        pad_offsets = (batch["attention_mask"] == 0).sum(dim=1)  # (bs,)

        # Pre-allocate per-batch
        batch_attn_idx = torch.zeros(bs, num_latent_pos, ATTN_TOPK, dtype=torch.long)
        batch_attn_val = torch.zeros(bs, num_latent_pos, ATTN_TOPK, dtype=torch.float32)
        batch_frac_input = torch.zeros(bs, num_latent_pos, dtype=torch.float32)
        batch_attn_to_lat = torch.zeros(bs, num_latent_pos, num_latent_pos, dtype=torch.float32)

        with torch.no_grad():
            # --- Encode ---
            eager_mask = make_eager_attention_mask(batch["attention_mask"], dtype=torch.bfloat16)
            outputs = model.codi(
                input_ids=batch["input_ids"], use_cache=True,
                output_hidden_states=True, past_key_values=None,
                attention_mask=eager_mask, output_attentions=True,
            )
            pkv = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # z0: attention from last pos (BOT) to input
            attn_stack = torch.stack(list(outputs.attentions), dim=0)  # (L, bs, H, seq, seq)
            attn_avg = attn_stack.mean(dim=(0, 2))  # (bs, seq, seq)
            attn_from_bot = attn_avg[:, -1, :].clone()  # (bs, seq)

            # Attention to input positions only (0..input_len-1)
            attn_to_input = attn_from_bot[:, :input_len].clone()
            pad_mask = (batch["attention_mask"] == 0)
            attn_to_input[pad_mask] = 0.0
            batch_frac_input[:, 0] = attn_to_input.sum(dim=-1).float()

            # Top-K from input (mask padding with -inf for topk)
            attn_for_topk = attn_to_input.clone()
            attn_for_topk[pad_mask] = -float('inf')
            k = min(ATTN_TOPK, input_len)
            top_vals, top_idx = torch.topk(attn_for_topk, k=k, dim=-1)

            for b in range(bs):
                offset = pad_offsets[b].item()
                batch_attn_idx[b, 0, :k] = (top_idx[b] - offset).clamp(min=0)
                batch_attn_val[b, 0, :k] = top_vals[b].float()

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # --- Latent iterations ---
            for it in range(inf_iters):
                outputs = model.codi(
                    inputs_embeds=latent_embd, use_cache=True,
                    output_hidden_states=True, past_key_values=pkv,
                    output_attentions=True,
                )
                pkv = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                lat_pos = it + 1  # z_{it+1}

                # Attention: (L, bs, H, 1, full_seq) for cached
                attn_stack = torch.stack(list(outputs.attentions), dim=0)
                attn_avg = attn_stack.mean(dim=(0, 2))  # (bs, 1, full_seq)
                attn_from_curr = attn_avg[:, 0, :].clone()  # (bs, full_seq)
                full_seq = attn_from_curr.shape[-1]

                # Fraction to input vs latent
                attn_input_part = attn_from_curr[:, :input_len].clone()
                attn_input_part[pad_mask] = 0.0
                batch_frac_input[:, lat_pos] = attn_input_part.sum(dim=-1).float()

                # Attention to each latent position (input_len onward)
                for prev_lat in range(min(lat_pos, full_seq - input_len)):
                    batch_attn_to_lat[:, lat_pos, prev_lat] = attn_from_curr[:, input_len + prev_lat].float()

                # Top-K from input positions
                attn_for_topk = attn_input_part.clone()
                attn_for_topk[pad_mask] = -float('inf')
                k = min(ATTN_TOPK, input_len)
                top_vals, top_idx = torch.topk(attn_for_topk, k=k, dim=-1)

                for b in range(bs):
                    offset = pad_offsets[b].item()
                    batch_attn_idx[b, lat_pos, :k] = (top_idx[b] - offset).clamp(min=0)
                    batch_attn_val[b, lat_pos, :k] = top_vals[b].float()

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

        # Save per-example data
        for b in range(bs):
            global_idx = step * data_args.batch_size + b
            if global_idx >= N:
                break
            all_attn_indices.append(batch_attn_idx[b].cpu())
            all_attn_values.append(batch_attn_val[b].cpu())
            all_frac_to_input.append(batch_frac_input[b].cpu())
            all_attn_to_latents.append(batch_attn_to_lat[b].cpu())
            offset = pad_offsets[b].item()
            all_input_tokens.append(batch["input_ids"][b, offset:].cpu().tolist())

        if (step + 1) % 3 == 0 or step == eval_step - 1:
            print(f"  Step {step+1}/{eval_step}")

    # --- Save ---
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/attention_tracking.pt"
    torch.save({
        "attn_top_indices": torch.stack(all_attn_indices),      # (N, 7, 20)
        "attn_top_values": torch.stack(all_attn_values),        # (N, 7, 20)
        "frac_to_input": torch.stack(all_frac_to_input),        # (N, 7)
        "attn_to_latents": torch.stack(all_attn_to_latents),    # (N, 7, 7)
        "input_token_ids": all_input_tokens,                     # list of lists
        "cot_strings": procedures,
        "questions": questions,
        "answers": answers,
    }, save_path)
    print(f"Saved {save_path} ({N} examples)")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    run(model_args, data_args, training_args)
