#!/usr/bin/env python3
"""
CPU script: loads latent_data.pt, early_stopping.json, decoded_latent.txt.
Computes all Tier 1 metrics, generates charts to outputs/charts/.

Experiments:
  B2. Early Stopping — accuracy vs. number of latent iterations
  B5. Logit Lens — per-layer token predictions for each latent
  A4. Attention Entropy — attention concentration per latent
  C3. Token Prediction Trajectory — how top-1 prediction evolves across latents

Usage:
    .venv/bin/python claude/scripts/analyze_all.py
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Style (same as analyze_latents.py)
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

PALETTE = plt.cm.tab10.colors
C_BLUE = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN = "#2ca02c"
C_RED = "#d62728"


def save_fig(fig, outdir, name):
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    fig.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close(fig)
    print(f"  Saved {name}")


# ===========================================================================
# B2. Early Stopping
# ===========================================================================
def plot_early_stopping(early_data, outdir, default_iters=6):
    """Line plot: accuracy vs. number of latent iterations."""
    iters = sorted(int(k) for k in early_data.keys())
    accs = [early_data[str(i)] for i in iters]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, accs, 'o-', color=C_BLUE, linewidth=2, markersize=8)

    # Annotate each point
    for i, acc in zip(iters, accs):
        offset = 1.5 if i != default_iters else -2.5
        weight = "bold" if i == default_iters else "normal"
        ax.annotate(f"{acc:.1f}%", (i, acc),
                    textcoords="offset points", xytext=(0, 10 + offset),
                    ha="center", fontsize=9, fontweight=weight)

    # Highlight default
    if default_iters in iters:
        idx = iters.index(default_iters)
        ax.plot(default_iters, accs[idx], 's', color=C_RED, markersize=12, zorder=5,
                label=f"Default ({default_iters} iters)")
        ax.legend()

    ax.set_xlabel("Number of Latent Iterations")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("B2. Early Stopping: Accuracy vs. Latent Iterations")
    ax.set_xticks(iters)
    ax.set_ylim(max(0, min(accs) - 10), 105)

    fig.tight_layout()
    save_fig(fig, outdir, "B2_early_stopping")


# ===========================================================================
# B5. Logit Lens
# ===========================================================================
def plot_logit_lens(data, tokenizer_decode, outdir):
    """
    Heatmap: (latent_index x layer_index), color = fraction where correct CoT number is top-1.
    Line plot: crystallization layer.
    """
    ids = data["logit_lens_top5_ids"]       # (N, 7, 13, 5)
    probs = data["logit_lens_top5_probs"]   # (N, 7, 13, 5)
    cot_steps = data["cot_steps"]
    correct = data["correct"]

    N, num_latents, num_layers, topk = ids.shape

    # --- Heatmap: fraction of correct examples where CoT step matches top-1 ---
    # Even latents (0,2,4,6) -> CoT steps (0,1,2,3)
    # For the heatmap, we check all latent positions
    # Map even latent positions to CoT step indices
    latent_to_cot = {0: 0, 2: 1, 4: 2, 6: 3}

    # Build heatmap: for each (latent, layer), fraction where top-1 matches CoT result
    heatmap = np.zeros((num_latents, num_layers))
    heatmap_count = np.zeros((num_latents, num_layers))

    for ex_idx in range(N):
        if not correct[ex_idx]:
            continue
        steps = cot_steps[ex_idx]
        for lat_idx in range(num_latents):
            cot_idx = latent_to_cot.get(lat_idx, None)
            if cot_idx is None or cot_idx >= len(steps):
                continue
            target_val = steps[cot_idx]
            for layer_idx in range(num_layers):
                heatmap_count[lat_idx, layer_idx] += 1
                top1_id = ids[ex_idx, lat_idx, layer_idx, 0].item()
                top1_str = tokenizer_decode(top1_id)
                try:
                    top1_val = float(top1_str.strip())
                    if abs(top1_val - target_val) < 1e-4:
                        heatmap[lat_idx, layer_idx] += 1
                except (ValueError, TypeError):
                    pass

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_frac = np.where(heatmap_count > 0, heatmap / heatmap_count, 0)

    # Plot heatmap (only even latents that have CoT mapping)
    even_latent_indices = [0, 2, 4, 6]
    even_latent_mask = [i for i in range(num_latents) if i in even_latent_indices and i < num_latents]
    heatmap_even = heatmap_frac[even_latent_mask, :]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(heatmap_even, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(len(even_latent_mask)))
    ax.set_yticklabels([f"z{i}" for i in even_latent_indices[:len(even_latent_mask)]])
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([f"L{i}" for i in range(num_layers)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Latent Position")
    ax.set_title("B5. Logit Lens: CoT Match Rate (top-1 = correct CoT number)")
    plt.colorbar(im, ax=ax, label="Fraction matching")

    # Annotate cells
    for yi, lat_i in enumerate(even_latent_mask):
        for xi in range(num_layers):
            val = heatmap_even[yi, xi]
            if val > 0.01:
                color = "white" if val > 0.5 else "black"
                ax.text(xi, yi, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.tight_layout()
    save_fig(fig, outdir, "B5_logit_lens_heatmap")

    # --- Crystallization layer: earliest layer where top-1 matches final layer's top-1 ---
    # For each example and latent, find the earliest layer where top-1 ID == final layer's top-1 ID
    crystal_layers = np.full((N, num_latents), num_layers - 1, dtype=float)

    for ex_idx in range(N):
        for lat_idx in range(num_latents):
            final_top1 = ids[ex_idx, lat_idx, -1, 0].item()
            for layer_idx in range(num_layers):
                if ids[ex_idx, lat_idx, layer_idx, 0].item() == final_top1:
                    crystal_layers[ex_idx, lat_idx] = layer_idx
                    break

    # Plot: mean crystallization layer per latent position
    mean_crystal = crystal_layers.mean(axis=0)
    std_crystal = crystal_layers.std(axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(num_latents)
    ax.errorbar(x, mean_crystal, yerr=std_crystal, fmt='o-', color=C_BLUE,
                linewidth=2, markersize=8, capsize=4)
    ax.set_xlabel("Latent Position (z0=initial, z1..z6=iterations)")
    ax.set_ylabel("Crystallization Layer")
    ax.set_title("B5. Logit Lens: Crystallization Layer per Latent")
    ax.set_xticks(x)
    ax.set_xticklabels([f"z{i}" for i in range(num_latents)])
    ax.set_ylim(-0.5, num_layers - 0.5)
    ax.invert_yaxis()

    fig.tight_layout()
    save_fig(fig, outdir, "B5_logit_lens_crystallization")

    return heatmap_frac, mean_crystal


# ===========================================================================
# A4. Attention Entropy
# ===========================================================================
def plot_attention_entropy(data, outdir):
    """
    Box plot: entropy distribution by latent index.
    Violin plot: correct vs incorrect.
    Scatter: entropy vs top-1 confidence.
    """
    entropy = data["attention_entropy"].numpy()       # (N, 7)
    entropy_input = data["attention_entropy_input_only"].numpy()  # (N, 7)
    correct = data["correct"].numpy()
    probs = data["logit_lens_top5_probs"].numpy()     # (N, 7, 13, 5)

    N, num_latents = entropy.shape

    # --- Box plot: entropy by latent index ---
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot([entropy[:, i] for i in range(num_latents)],
                    labels=[f"z{i}" for i in range(num_latents)],
                    patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)
    ax.set_xlabel("Latent Position")
    ax.set_ylabel("Attention Entropy (nats)")
    ax.set_title("A4. Attention Entropy per Latent Position")
    fig.tight_layout()
    save_fig(fig, outdir, "A4_entropy_boxplot")

    # --- Violin plot: correct vs incorrect ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (mask, label) in zip(axes, [(correct, "Correct"), (~correct, "Incorrect")]):
        subset = entropy[mask]
        if len(subset) == 0:
            ax.set_title(f"{label} (n=0)")
            continue
        vp = ax.violinplot([subset[:, i] for i in range(num_latents)],
                           positions=range(num_latents), showmedians=True)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(PALETTE[i % len(PALETTE)])
            body.set_alpha(0.6)
        ax.set_xticks(range(num_latents))
        ax.set_xticklabels([f"z{i}" for i in range(num_latents)])
        ax.set_xlabel("Latent Position")
        ax.set_title(f"{label} (n={mask.sum()})")
    axes[0].set_ylabel("Attention Entropy (nats)")
    fig.suptitle("A4. Attention Entropy: Correct vs Incorrect", y=1.02)
    fig.tight_layout()
    save_fig(fig, outdir, "A4_entropy_violin_correct_incorrect")

    # --- Scatter: entropy vs top-1 confidence (final layer) ---
    # Use the final layer's top-1 probability as confidence
    final_layer_top1_prob = probs[:, :, -1, 0]  # (N, 7)

    fig, ax = plt.subplots(figsize=(6, 5))
    # Plot for each latent position
    for lat_idx in [0, 2, 4, 6]:  # even latents only for clarity
        if lat_idx >= num_latents:
            continue
        ax.scatter(entropy[:, lat_idx], final_layer_top1_prob[:, lat_idx],
                   alpha=0.15, s=10, color=PALETTE[lat_idx], label=f"z{lat_idx}")
    ax.set_xlabel("Attention Entropy (nats)")
    ax.set_ylabel("Top-1 Confidence (final layer)")
    ax.set_title("A4. Entropy vs. Prediction Confidence")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, outdir, "A4_entropy_vs_confidence")

    return entropy


# ===========================================================================
# C3. Token Prediction Trajectory
# ===========================================================================
def parse_decoded_latent(filepath):
    """Parse decoded_latent.txt to extract top-1 even-latent predictions per example."""
    examples = []
    current_preds = {}
    current_answer = None
    current_pred_text = None

    with open(filepath, "r") as f:
        lines = f.read().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Question") and "..." in line:
            # Save previous
            if current_preds:
                examples.append({
                    "preds": current_preds,
                    "answer": current_answer,
                    "pred_text": current_pred_text,
                })
            current_preds = {}
            current_answer = None
            current_pred_text = None
            i += 1
            continue

        if line.startswith("CoT="):
            m = re.match(r"CoT=.+,\s*Answer=(.+)", line)
            if m:
                try:
                    current_answer = float(m.group(1).strip())
                except ValueError:
                    pass
            i += 1
            continue

        # "decoded 0th latent (top5): ['token1', 'token2', ...]"
        m = re.match(r"decoded (\d+)(?:th|st|nd|rd) latent \(top\d+\): \[(.+)\]", line)
        if m:
            idx = int(m.group(1))
            tokens = re.findall(r"'([^']*)'", m.group(2))
            if tokens:
                current_preds[idx] = tokens[0].strip()
            i += 1
            continue

        if line.startswith("Model Prediction:"):
            current_pred_text = line.replace("Model Prediction:", "").strip()
            i += 1
            continue

        i += 1

    if current_preds:
        examples.append({
            "preds": current_preds,
            "answer": current_answer,
            "pred_text": current_pred_text,
        })

    return examples


def classify_trajectory(preds):
    """Classify prediction trajectory from even latents z0, z2, z4, z6.
    Returns one of: 'Immediate', 'Early converge', 'Late converge', 'No converge'
    """
    z = [preds.get(i, None) for i in [0, 2, 4, 6]]
    # Filter None
    z = [x for x in z if x is not None]
    if len(z) < 2:
        return "Too few"

    if len(set(z)) == 1:
        return "Immediate"

    # Check from end
    if len(z) >= 3 and z[-1] == z[-2] == z[-3]:
        # z2=z4=z6 (regardless of z0)
        return "Early converge"
    if len(z) >= 2 and z[-1] == z[-2]:
        return "Late converge"

    return "No converge"


def plot_trajectory(traj_examples, outdir):
    """
    Bar chart: fraction in each trajectory category.
    Grouped bar: accuracy by trajectory category.
    Sankey-like: transition diagram.
    """
    categories = Counter()
    category_correct = Counter()
    category_total = Counter()

    for ex in traj_examples:
        cat = classify_trajectory(ex["preds"])
        if cat == "Too few":
            continue
        categories[cat] += 1

        # Check accuracy
        if ex["answer"] is not None and ex["pred_text"]:
            nums = re.findall(r"-?\d+\.?\d*", ex["pred_text"])
            if nums:
                pred_val = float(nums[-1])
                is_correct = abs(pred_val - ex["answer"]) < 1e-4
                category_total[cat] += 1
                if is_correct:
                    category_correct[cat] += 1

    order = ["Immediate", "Early converge", "Late converge", "No converge"]
    cats_present = [c for c in order if c in categories]

    # --- Bar chart: fraction in each category ---
    total = sum(categories.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(cats_present))
    fracs = [categories[c] / total for c in cats_present]
    bars = ax.bar(x, fracs, 0.55, color=[PALETTE[i] for i in range(len(cats_present))])
    ax.set_xticks(x)
    ax.set_xticklabels(cats_present, rotation=15, ha="right")
    ax.set_ylabel("Fraction of Examples")
    ax.set_title("C3. Token Prediction Trajectory Categories")
    ax.set_ylim(0, 1.1)
    for i, bar in enumerate(bars):
        n = categories[cats_present[i]]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{fracs[i]*100:.1f}%\n(n={n})", ha="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, outdir, "C3_trajectory_categories")

    # --- Grouped bar: accuracy by category ---
    fig, ax = plt.subplots(figsize=(6, 4))
    accs = []
    for c in cats_present:
        if category_total[c] > 0:
            accs.append(category_correct[c] / category_total[c])
        else:
            accs.append(0)
    bars = ax.bar(x, accs, 0.55, color=[PALETTE[i] for i in range(len(cats_present))])
    ax.set_xticks(x)
    ax.set_xticklabels(cats_present, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("C3. Accuracy by Trajectory Category")
    ax.set_ylim(0, 1.15)
    for i, bar in enumerate(bars):
        n = category_total[cats_present[i]]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{accs[i]*100:.1f}%\n(n={n})", ha="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, outdir, "C3_trajectory_accuracy")

    # --- Transition diagram (simplified as grouped bar of top-1 token changes) ---
    # Count transitions z0->z2, z2->z4, z4->z6: same vs different
    transitions = {"z0->z2": [0, 0], "z2->z4": [0, 0], "z4->z6": [0, 0]}
    pairs = [(0, 2, "z0->z2"), (2, 4, "z2->z4"), (4, 6, "z4->z6")]
    for ex in traj_examples:
        for a, b, label in pairs:
            if a in ex["preds"] and b in ex["preds"]:
                if ex["preds"][a] == ex["preds"][b]:
                    transitions[label][0] += 1  # same
                else:
                    transitions[label][1] += 1  # different

    fig, ax = plt.subplots(figsize=(6, 4))
    labels_t = list(transitions.keys())
    same_fracs = []
    diff_fracs = []
    for label in labels_t:
        s, d = transitions[label]
        tot = s + d
        same_fracs.append(s / tot if tot > 0 else 0)
        diff_fracs.append(d / tot if tot > 0 else 0)

    x_t = np.arange(len(labels_t))
    w = 0.35
    ax.bar(x_t - w/2, same_fracs, w, label="Same prediction", color=C_BLUE)
    ax.bar(x_t + w/2, diff_fracs, w, label="Different prediction", color=C_ORANGE)
    ax.set_xticks(x_t)
    ax.set_xticklabels(labels_t)
    ax.set_ylabel("Fraction")
    ax.set_title("C3. Prediction Stability Between Latent Positions")
    ax.set_ylim(0, 1.15)
    ax.legend()

    for i, label in enumerate(labels_t):
        s, d = transitions[label]
        ax.text(x_t[i] - w/2, same_fracs[i] + 0.02, f"{same_fracs[i]*100:.0f}%",
                ha="center", fontsize=8)
        ax.text(x_t[i] + w/2, diff_fracs[i] + 0.02, f"{diff_fracs[i]*100:.0f}%",
                ha="center", fontsize=8)

    fig.tight_layout()
    save_fig(fig, outdir, "C3_trajectory_transitions")

    return categories, category_correct, category_total


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Tier 1 Latent Analysis: all charts")
    parser.add_argument("--latent-data", default="outputs/latent_data.pt",
                        help="Path to latent_data.pt from latent_analysis.py")
    parser.add_argument("--early-stopping", default="outputs/early_stopping.json",
                        help="Path to early_stopping.json")
    parser.add_argument("--decoded-latent", default="outputs/decoded_latent.txt",
                        help="Path to decoded_latent.txt from probe_latent_token.py")
    parser.add_argument("--outdir", "-o", default="outputs/charts")
    parser.add_argument("--tokenizer", default="gpt2",
                        help="Tokenizer name for decoding logit lens IDs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load tokenizer for decoding
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    def decode_id(tok_id):
        try:
            return tokenizer.decode([int(tok_id)])
        except:
            return f"<{tok_id}>"

    # -----------------------------------------------------------------------
    # B2. Early Stopping
    # -----------------------------------------------------------------------
    if os.path.exists(args.early_stopping):
        print("\n=== B2. Early Stopping ===")
        with open(args.early_stopping) as f:
            early_data = json.load(f)
        plot_early_stopping(early_data, args.outdir)
        print(f"  Results: {early_data}")
    else:
        print(f"  [SKIP] {args.early_stopping} not found")

    # -----------------------------------------------------------------------
    # B5. Logit Lens + A4. Attention Entropy
    # -----------------------------------------------------------------------
    if os.path.exists(args.latent_data):
        data = torch.load(args.latent_data, map_location="cpu", weights_only=False)
        print(f"\n  Loaded latent_data.pt: {list(data.keys())}")
        print(f"  Shapes: ids={data['logit_lens_top5_ids'].shape}, "
              f"entropy={data['attention_entropy'].shape}, "
              f"correct={data['correct'].shape}")

        print("\n=== B5. Logit Lens ===")
        plot_logit_lens(data, decode_id, args.outdir)

        print("\n=== A4. Attention Entropy ===")
        plot_attention_entropy(data, args.outdir)
    else:
        print(f"  [SKIP] {args.latent_data} not found")

    # -----------------------------------------------------------------------
    # C3. Token Prediction Trajectory
    # -----------------------------------------------------------------------
    if os.path.exists(args.decoded_latent):
        print("\n=== C3. Token Prediction Trajectory ===")
        traj_examples = parse_decoded_latent(args.decoded_latent)
        print(f"  Parsed {len(traj_examples)} examples from decoded_latent.txt")
        cats, cat_correct, cat_total = plot_trajectory(traj_examples, args.outdir)
        print(f"  Categories: {dict(cats)}")
        for c in cats:
            n = cat_total[c]
            acc = cat_correct[c] / n * 100 if n > 0 else 0
            print(f"    {c}: {cats[c]} examples, accuracy={acc:.1f}% ({cat_correct[c]}/{n})")
    else:
        print(f"  [SKIP] {args.decoded_latent} not found")

    print("\nDone. Charts saved to", args.outdir)


if __name__ == "__main__":
    main()
