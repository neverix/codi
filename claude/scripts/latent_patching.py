#!/usr/bin/env python3
"""
GPU script: causal intervention experiments on latent tokens.

Experiments:
  1. Baseline (normal inference)
  2. Single-position ablation: zero latent at position i (i=0..6)
  3. Token forcing correct: at even position i, replace with embed(correct CoT result)
  4. Token forcing wrong: at even position i, replace with embed(correct+1)
  5. Frozen latent: repeat z0 at every position (no iteration refinement)

Saves to outputs/latent_patching.json.
"""

import math
import re
import os
import json

import torch
import transformers
from torch.nn import functional as F

from peft import LoraConfig, TaskType
from datasets import load_dataset
from safetensors.torch import load_file

from src.model import CODI, ModelArguments, DataArguments, TrainingArguments

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


def parse_cot_operations(cot_str):
    """Parse CoT like <<16-3-4=9>><<9*2=18>> into operations with operands and results."""
    operations = []
    for block in re.findall(r"<<(.+?)>>", cot_str):
        parts = block.split("=")
        if len(parts) < 2:
            continue
        expression = "=".join(parts[:-1])
        operands = [float(x) for x in re.findall(r'-?\d+\.?\d*', expression)]
        try:
            result = float(parts[-1])
        except ValueError:
            continue
        operations.append({"operands": operands, "result": result})
    return operations


def number_to_embedding(number, tokenizer, embed_fn, dev):
    """Convert a number to an embedding vector (average if multi-token)."""
    num_str = str(int(number)) if number == int(number) else str(number)
    token_ids = tokenizer.encode(num_str, add_special_tokens=False)
    if not token_ids:
        return None
    ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=dev)
    embeds = embed_fn(ids_tensor)  # (num_tokens, dim)
    return embeds.mean(dim=0)  # average for multi-token numbers


def precompute_forcing_embeddings(procedures, position, tokenizer, embed_fn, dev, offset=0):
    """For each example, compute the token embedding to force at an even latent position.

    position: even latent index (0,2,4,6) → CoT step index (0,1,2,3)
    offset: 0 for correct, +1 for wrong (correct+1)

    Returns: list of (embedding_or_None, target_number_or_None) per example
    """
    cot_step_idx = position // 2
    results = []
    for cot_str in procedures:
        ops = parse_cot_operations(cot_str)
        if cot_step_idx < len(ops):
            target = ops[cot_step_idx]["result"] + offset
            emb = number_to_embedding(target, tokenizer, embed_fn, dev)
            results.append((emb, target))
        else:
            results.append((None, None))
    return results


def run_eval(model, tokenizer, question_data, answer, training_args, gen_kwargs,
             data_args, intervention=None, procedures=None):
    """Run full eval with optional intervention.

    intervention: dict with:
      type: "ablate" | "force_correct" | "force_wrong" | "frozen"
      position: int (for ablate/force, which latent position)
      embeddings: list of (emb_or_None, val) per example (for force)
    """
    model.eval()
    ans_pred_list = []
    embed_fn = model.get_embd(model.codi, model.model_name)
    inf_iters = training_args.inf_latent_iterations

    for step, batch in enumerate(question_data):
        bs = batch["input_ids"].size(0)
        global_start = step * data_args.batch_size

        with torch.no_grad():
            pkv = None
            eager_mask = make_eager_attention_mask(batch["attention_mask"], dtype=torch.bfloat16)
            outputs = model.codi(
                input_ids=batch["input_ids"], use_cache=True,
                output_hidden_states=True, past_key_values=pkv,
                attention_mask=eager_mask, output_attentions=False,
            )
            pkv = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # z0 is now the projected embedding — apply intervention
            frozen_z0 = latent_embd.clone() if (intervention and intervention["type"] == "frozen") else None

            latent_embd = _apply_intervention(latent_embd, 0, intervention, global_start, bs, embed_fn, frozen_z0)

            for it in range(inf_iters):
                outputs = model.codi(
                    inputs_embeds=latent_embd, use_cache=True,
                    output_hidden_states=True, past_key_values=pkv,
                    output_attentions=False,
                )
                pkv = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

                lat_pos = it + 1
                latent_embd = _apply_intervention(latent_embd, lat_pos, intervention, global_start, bs, embed_fn, frozen_z0)

            # --- Generate answer ---
            if training_args.remove_eos:
                eot_emb = embed_fn(torch.tensor([model.eot_id], dtype=torch.long, device='cuda')).unsqueeze(0)
            else:
                eot_emb = embed_fn(torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0)
            eot_emb = eot_emb.expand(bs, -1, -1)

            output = eot_emb
            finished = torch.zeros(bs, dtype=torch.bool, device="cuda")
            pred_tokens = [[] for _ in range(bs)]

            for i in range(gen_kwargs["max_new_tokens"]):
                out = model.codi(
                    inputs_embeds=output, output_hidden_states=False,
                    attention_mask=None, use_cache=True,
                    output_attentions=False, past_key_values=pkv,
                )
                pkv = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]

                if training_args.greedy:
                    next_ids = torch.argmax(logits, dim=-1).view(-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        tk_vals, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        logits[logits < tk_vals[:, -1:]] = -float("inf")
                    if gen_kwargs["top_p"] < 1.0:
                        sorted_l, sorted_i = torch.sort(logits, descending=True, dim=-1)
                        cum_p = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                        remove = cum_p > gen_kwargs["top_p"]
                        if remove.any():
                            remove = remove.roll(1, dims=-1)
                            remove[:, 0] = False
                        for b in range(logits.size(0)):
                            logits[b, sorted_i[b, remove[b]]] = -float("inf")
                    probs = F.softmax(logits, dim=-1)
                    next_ids = torch.multinomial(probs, num_samples=1).view(-1)

                for b in range(bs):
                    if not finished[b]:
                        pred_tokens[b].append(next_ids[b].item())
                        if next_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True
                if finished.all():
                    break
                output = embed_fn(next_ids).unsqueeze(1).to(device)

            for b in range(bs):
                decoded = tokenizer.decode(pred_tokens[b], skip_special_tokens=True)
                ans_pred_list.append(extract_answer_number(decoded))

    # Accuracy
    correct = sum(1 for p, g in zip(ans_pred_list, answer) if p == g)
    return correct / len(answer), ans_pred_list


def _apply_intervention(latent_embd, lat_pos, intervention, global_start, bs, embed_fn, frozen_z0):
    """Apply intervention to latent embedding at position lat_pos."""
    if intervention is None:
        return latent_embd

    itype = intervention["type"]

    if itype == "ablate" and intervention["position"] == lat_pos:
        return torch.zeros_like(latent_embd)

    if itype in ("force_correct", "force_wrong") and intervention["position"] == lat_pos:
        embeddings = intervention["embeddings"]
        result = latent_embd.clone()
        for b in range(bs):
            ex_idx = global_start + b
            if ex_idx < len(embeddings):
                emb, _ = embeddings[ex_idx]
                if emb is not None:
                    result[b, 0, :] = emb.to(result.dtype)
        return result

    if itype == "frozen" and frozen_z0 is not None:
        return frozen_z0.clone()

    return latent_embd


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- Model setup ---
    if "gpt2" in model_args.model_name_or_path.lower():
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif any(n in model_args.model_name_or_path.lower() for n in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    else:
        raise ValueError(f"Unsupported: {model_args.model_name_or_path}")

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

    # --- Dataset ---
    dataset = load_dataset(data_args.data_name)
    test_set = dataset['test']
    questions, answer, procedures = [], [], []
    for ex in test_set:
        questions.append(ex['question'].strip().replace('  ', ' '))
        answer.append(float(ex['answer'].replace(',', '')))
        procedures.append(ex['cot'])

    N = len(questions)
    eval_step = math.ceil(N / data_args.batch_size)
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

    gen_kwargs = {"max_new_tokens": 256, "temperature": 0.1, "top_k": 40, "top_p": 0.95, "do_sample": True}
    embed_fn = model.get_embd(model.codi, model.model_name)
    inf_iters = training_args.inf_latent_iterations

    results = {}

    # --- 1. Baseline ---
    print("\n=== Baseline ===")
    acc, _ = run_eval(model, tokenizer, question_data, answer, training_args, gen_kwargs, data_args)
    results["baseline"] = round(100 * acc, 2)
    print(f"  Baseline: {results['baseline']}%")

    # --- 2. Single-position ablation ---
    print("\n=== Ablation ===")
    results["ablation"] = {}
    for pos in range(1 + inf_iters):
        print(f"  Ablating z{pos}...")
        acc, _ = run_eval(model, tokenizer, question_data, answer, training_args, gen_kwargs, data_args,
                          intervention={"type": "ablate", "position": pos})
        results["ablation"][pos] = round(100 * acc, 2)
        print(f"    z{pos} ablated: {results['ablation'][pos]}%")

    # --- 3. Token forcing correct (even positions) ---
    print("\n=== Token forcing correct ===")
    results["force_correct"] = {}
    for pos in [0, 2, 4, 6]:
        if pos > inf_iters:
            break
        embeddings = precompute_forcing_embeddings(procedures, pos, tokenizer, embed_fn, device, offset=0)
        valid = sum(1 for e, _ in embeddings if e is not None)
        print(f"  Forcing correct at z{pos} ({valid}/{N} have CoT step)...")
        acc, _ = run_eval(model, tokenizer, question_data, answer, training_args, gen_kwargs, data_args,
                          intervention={"type": "force_correct", "position": pos, "embeddings": embeddings})
        results["force_correct"][pos] = round(100 * acc, 2)
        print(f"    z{pos} force correct: {results['force_correct'][pos]}%")

    # --- 4. Token forcing wrong (even positions) ---
    print("\n=== Token forcing wrong ===")
    results["force_wrong"] = {}
    for pos in [0, 2, 4, 6]:
        if pos > inf_iters:
            break
        embeddings = precompute_forcing_embeddings(procedures, pos, tokenizer, embed_fn, device, offset=1)
        valid = sum(1 for e, _ in embeddings if e is not None)
        print(f"  Forcing wrong at z{pos} ({valid}/{N} have CoT step)...")
        acc, _ = run_eval(model, tokenizer, question_data, answer, training_args, gen_kwargs, data_args,
                          intervention={"type": "force_wrong", "position": pos, "embeddings": embeddings})
        results["force_wrong"][pos] = round(100 * acc, 2)
        print(f"    z{pos} force wrong: {results['force_wrong'][pos]}%")

    # --- 5. Frozen latent ---
    print("\n=== Frozen latent ===")
    acc, _ = run_eval(model, tokenizer, question_data, answer, training_args, gen_kwargs, data_args,
                      intervention={"type": "frozen"})
    results["frozen"] = round(100 * acc, 2)
    print(f"  Frozen z0: {results['frozen']}%")

    # --- Save ---
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/latent_patching.json"
    # Convert int keys to strings for JSON
    results_json = {
        "baseline": results["baseline"],
        "ablation": {str(k): v for k, v in results["ablation"].items()},
        "force_correct": {str(k): v for k, v in results["force_correct"].items()},
        "force_wrong": {str(k): v for k, v in results["force_wrong"].items()},
        "frozen": results["frozen"],
    }
    with open(save_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved to {save_path}")
    print(json.dumps(results_json, indent=2))


if __name__ == "__main__":
    main()
