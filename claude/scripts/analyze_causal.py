#!/usr/bin/env python3
"""
CPU script: analyze attention tracking + latent patching results.
Generates charts to outputs/charts/.

Usage:
    .venv/bin/python claude/scripts/analyze_causal.py
"""

import os
import re
import json
import argparse
from collections import Counter

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 12, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})
PALETTE = plt.cm.tab10.colors
C_BLUE, C_ORANGE, C_GREEN, C_RED = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"


def save_fig(fig, outdir, name):
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close(fig)
    print(f"  Saved {name}")


# ===========================================================================
# CoT parsing
# ===========================================================================
def parse_cot_operations(cot_str):
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


def find_number_positions(token_ids, tokenizer):
    """Map number values to their token positions in the input."""
    vocab_size = tokenizer.vocab_size
    positions = {}
    for pos, tid in enumerate(token_ids):
        if tid >= vocab_size:
            continue  # skip special tokens (BOT, EOT, PAD)
        try:
            decoded = tokenizer.decode([tid]).strip()
        except Exception:
            continue
        nums = re.findall(r'-?\d+\.?\d*', decoded)
        for num_str in nums:
            val = float(num_str)
            if val not in positions:
                positions[val] = []
            positions[val].append(pos)
    return positions


# ===========================================================================
# Attention-Input Tracking Analysis
# ===========================================================================
def analyze_attention_tracking(data, tokenizer, outdir):
    attn_idx = data["attn_top_indices"]     # (N, 7, 20)
    attn_val = data["attn_top_values"]      # (N, 7, 20)
    frac_input = data["frac_to_input"]      # (N, 7)
    attn_to_lat = data["attn_to_latents"]   # (N, 7, 7)
    input_tokens = data["input_token_ids"]  # list of lists
    cot_strings = data["cot_strings"]
    questions = data["questions"]

    N = len(input_tokens)
    num_latents = attn_idx.shape[1]
    even_positions = [0, 2, 4, 6]
    latent_to_cot = {0: 0, 2: 1, 4: 2, 6: 3}

    # --- Operand match analysis ---
    # For each even latent, check if CoT step operands are in top-K attended
    topk_values = [1, 3, 5, 10, 20]
    # match_rates[k][lat_pos] = (hits, attempts)
    match_rates = {k: {pos: [0, 0] for pos in even_positions} for k in topk_values}
    # Also track source vs intermediate operands
    source_match = {k: [0, 0] for k in topk_values}
    intermediate_match = {k: [0, 0] for k in topk_values}

    for ex_idx in range(N):
        ops = parse_cot_operations(cot_strings[ex_idx])
        num_pos = find_number_positions(input_tokens[ex_idx], tokenizer)

        prev_results = set()
        for lat_pos in even_positions:
            cot_idx = latent_to_cot[lat_pos]
            if cot_idx >= len(ops) or lat_pos >= num_latents:
                continue
            op = ops[cot_idx]

            for operand in op["operands"]:
                operand_positions = set(num_pos.get(operand, []))
                is_source = operand not in prev_results

                for k in topk_values:
                    top_k_set = set(attn_idx[ex_idx, lat_pos, :k].tolist())
                    hit = bool(operand_positions & top_k_set)
                    match_rates[k][lat_pos][1] += 1
                    if hit:
                        match_rates[k][lat_pos][0] += 1
                    if is_source:
                        source_match[k][1] += 1
                        if hit:
                            source_match[k][0] += 1
                    else:
                        intermediate_match[k][1] += 1
                        if hit:
                            intermediate_match[k][0] += 1

            prev_results.add(op["result"])

    # --- Chart: Operand match rate by latent position ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(even_positions))
    width = 0.15
    for i, k in enumerate([1, 5, 10, 20]):
        rates = []
        for pos in even_positions:
            h, a = match_rates[k][pos]
            rates.append(h / a if a > 0 else 0)
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, rates, width, label=f"Top-{k}", color=PALETTE[i])
        for j, bar in enumerate(bars):
            if rates[j] > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{rates[j]*100:.0f}%", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"z{p}" for p in even_positions])
    ax.set_ylabel("Operand Match Rate")
    ax.set_xlabel("Latent Position")
    ax.set_title("Are CoT Operands in Top-Attended Input Tokens?")
    ax.set_ylim(0, 1.15)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, outdir, "attn_operand_match_by_position")

    # --- Chart: Source vs intermediate operand match ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ks_plot = [1, 3, 5, 10, 20]
    src_rates = [source_match[k][0] / source_match[k][1] if source_match[k][1] > 0 else 0 for k in ks_plot]
    int_rates = [intermediate_match[k][0] / intermediate_match[k][1] if intermediate_match[k][1] > 0 else 0 for k in ks_plot]
    x = np.arange(len(ks_plot))
    w = 0.35
    ax.bar(x - w/2, src_rates, w, label="Source (from question)", color=C_BLUE)
    ax.bar(x + w/2, int_rates, w, label="Intermediate (prev step result)", color=C_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks_plot])
    ax.set_ylabel("Match Rate")
    ax.set_title("Source vs Intermediate Operand Attention")
    ax.set_ylim(0, 1.15)
    ax.legend()
    for i in range(len(ks_plot)):
        ax.text(x[i] - w/2, src_rates[i] + 0.02, f"{src_rates[i]*100:.0f}%", ha="center", fontsize=7)
        ax.text(x[i] + w/2, int_rates[i] + 0.02, f"{int_rates[i]*100:.0f}%", ha="center", fontsize=7)
    fig.tight_layout()
    save_fig(fig, outdir, "attn_source_vs_intermediate")

    # --- Chart: Fraction of attention to input vs latent positions ---
    fig, ax = plt.subplots(figsize=(6, 4))
    frac_np = frac_input.numpy()  # (N, 7)
    means = frac_np.mean(axis=0)
    stds = frac_np.std(axis=0)
    x = np.arange(num_latents)
    ax.bar(x, means, 0.6, yerr=stds, color=C_BLUE, alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"z{i}" for i in range(num_latents)])
    ax.set_ylabel("Fraction of Attention to Input")
    ax.set_xlabel("Latent Position")
    ax.set_title("How Much Attention Goes to Input vs Latent Positions?")
    ax.set_ylim(0, 1.05)
    for i in range(num_latents):
        ax.text(i, means[i] + stds[i] + 0.02, f"{means[i]*100:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, outdir, "attn_fraction_to_input")

    # --- Chart: Attention to latent positions heatmap ---
    # attn_to_lat: (N, 7, 7) — for z_i, how much attention goes to z_j (j < i)
    attn_lat_mean = attn_to_lat.numpy().mean(axis=0)  # (7, 7)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_lat_mean[1:, :], aspect='auto', cmap='Blues', vmin=0)
    ax.set_yticks(range(num_latents - 1))
    ax.set_yticklabels([f"z{i}" for i in range(1, num_latents)])
    ax.set_xticks(range(num_latents))
    ax.set_xticklabels([f"z{i}" for i in range(num_latents)])
    ax.set_xlabel("Attended-to Latent Position")
    ax.set_ylabel("Current Latent Position")
    ax.set_title("Inter-Latent Attention Patterns")
    plt.colorbar(im, ax=ax, label="Attention weight")
    for yi in range(num_latents - 1):
        for xi in range(num_latents):
            val = attn_lat_mean[yi + 1, xi]
            if val > 0.005:
                ax.text(xi, yi, f"{val:.3f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    save_fig(fig, outdir, "attn_inter_latent_heatmap")

    # --- Print summary ---
    print("\n  Operand match rates (top-5):")
    for pos in even_positions:
        h, a = match_rates[5][pos]
        print(f"    z{pos} → CoT step {pos//2}: {h}/{a} = {h/a*100:.1f}%" if a > 0 else f"    z{pos}: no data")
    print(f"  Source operands in top-5: {source_match[5][0]}/{source_match[5][1]} = "
          f"{source_match[5][0]/source_match[5][1]*100:.1f}%" if source_match[5][1] > 0 else "")
    print(f"  Intermediate operands in top-5: {intermediate_match[5][0]}/{intermediate_match[5][1]} = "
          f"{intermediate_match[5][0]/intermediate_match[5][1]*100:.1f}%" if intermediate_match[5][1] > 0 else "")


# ===========================================================================
# Latent Patching Analysis
# ===========================================================================
def analyze_patching(patch_data, outdir):
    baseline = patch_data["baseline"]

    # --- Ablation chart ---
    ablation = patch_data["ablation"]
    positions = sorted(int(k) for k in ablation.keys())
    abl_accs = [ablation[str(p)] for p in positions]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(positions))
    bars = ax.bar(x, abl_accs, 0.55, color=[PALETTE[i % len(PALETTE)] for i in range(len(positions))])
    ax.axhline(y=baseline, color=C_RED, linestyle='--', linewidth=2, label=f"Baseline ({baseline}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"z{p}" for p in positions])
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Ablated Position")
    ax.set_title("Latent Ablation: Accuracy When Zeroing Each Position")
    ax.legend()
    for i, bar in enumerate(bars):
        delta = abl_accs[i] - baseline
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{abl_accs[i]:.1f}%\n({delta:+.1f})", ha="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, outdir, "patch_ablation")

    # --- Token forcing chart ---
    fc = patch_data.get("force_correct", {})
    fw = patch_data.get("force_wrong", {})
    if fc or fw:
        even_pos = sorted(set(int(k) for k in list(fc.keys()) + list(fw.keys())))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(even_pos))
        w = 0.25

        fc_vals = [fc.get(str(p), baseline) for p in even_pos]
        fw_vals = [fw.get(str(p), baseline) for p in even_pos]

        ax.bar(x - w, fc_vals, w, label="Force correct", color=C_GREEN)
        ax.bar(x, [baseline] * len(even_pos), w, label="Baseline", color=C_BLUE, alpha=0.5)
        ax.bar(x + w, fw_vals, w, label="Force wrong (+1)", color=C_RED)

        ax.set_xticks(x)
        ax.set_xticklabels([f"z{p}" for p in even_pos])
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Forced Position")
        ax.set_title("Token Forcing: Correct vs Wrong Number at Each Even Latent")
        ax.legend()
        for i in range(len(even_pos)):
            for val, offset in [(fc_vals[i], -w), (fw_vals[i], w)]:
                delta = val - baseline
                ax.text(x[i] + offset, val + 0.5, f"{val:.1f}%\n({delta:+.1f})",
                        ha="center", fontsize=7)
        fig.tight_layout()
        save_fig(fig, outdir, "patch_token_forcing")

    # --- Summary with frozen ---
    frozen = patch_data.get("frozen", None)
    fig, ax = plt.subplots(figsize=(8, 4))
    conditions = ["Baseline"]
    values = [baseline]
    colors = [C_BLUE]

    # Add most impactful ablation
    worst_abl_pos = min(ablation, key=lambda k: ablation[k])
    conditions.append(f"Ablate z{worst_abl_pos}")
    values.append(ablation[worst_abl_pos])
    colors.append(C_RED)

    # Best force correct
    if fc:
        best_fc_pos = max(fc, key=lambda k: fc[k])
        conditions.append(f"Force correct z{best_fc_pos}")
        values.append(fc[best_fc_pos])
        colors.append(C_GREEN)

    # Worst force wrong
    if fw:
        worst_fw_pos = min(fw, key=lambda k: fw[k])
        conditions.append(f"Force wrong z{worst_fw_pos}")
        values.append(fw[worst_fw_pos])
        colors.append(C_ORANGE)

    if frozen is not None:
        conditions.append("Frozen z0")
        values.append(frozen)
        colors.append(PALETTE[4])

    x = np.arange(len(conditions))
    bars = ax.bar(x, values, 0.55, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Latent Patching: Key Results Summary")
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{values[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, outdir, "patch_summary")

    # Print
    print(f"\n  Baseline: {baseline}%")
    print(f"  Ablation: {ablation}")
    print(f"  Force correct: {fc}")
    print(f"  Force wrong: {fw}")
    print(f"  Frozen: {frozen}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-data", default="outputs/attention_tracking.pt")
    parser.add_argument("--patch-data", default="outputs/latent_patching.json")
    parser.add_argument("--outdir", "-o", default="outputs/charts")
    parser.add_argument("--tokenizer", default="gpt2")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)

    if os.path.exists(args.attn_data):
        print("=== Attention-Input Tracking ===")
        data = torch.load(args.attn_data, map_location="cpu", weights_only=False)
        print(f"  Loaded: {len(data['input_token_ids'])} examples")
        analyze_attention_tracking(data, tokenizer, args.outdir)
    else:
        print(f"  [SKIP] {args.attn_data} not found")

    if os.path.exists(args.patch_data) and os.path.getsize(args.patch_data) > 0:
        print("\n=== Latent Patching ===")
        with open(args.patch_data) as f:
            patch_data = json.load(f)
        analyze_patching(patch_data, args.outdir)
    else:
        print(f"  [SKIP] {args.patch_data} not found")

    print(f"\nDone. Charts in {args.outdir}")


if __name__ == "__main__":
    main()
