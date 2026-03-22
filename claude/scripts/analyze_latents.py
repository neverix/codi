#!/usr/bin/env python3
"""
Analyze CODI latent token predictions from decoded_latent.txt.
Produces publication-quality PDF/PNG charts.

Usage:
    .venv/bin/python claude/scripts/analyze_latents.py [--input FILE] [--outdir DIR]
"""

import re
import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Style
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

# Colorblind-friendly palette (tab10 subset)
C_NUM   = "#1f77b4"   # blue
C_OP    = "#ff7f0e"   # orange
C_OTHER = "#2ca02c"   # green
C_ACCENT = "#d62728"  # red
PALETTE = plt.cm.tab10.colors

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class LatentStep:
    index: int                    # 0..7 (or "before_answer")
    attended_tokens: List[str]
    decoded_tokens: List[str]

@dataclass
class Example:
    question_id: int
    question: str
    cot: str
    answer: float
    latent_steps: List[LatentStep] = field(default_factory=list)
    prediction: str = ""
    predicted_answer: Optional[float] = None
    correct: bool = False

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_file(filepath: str) -> List[Example]:
    """Parse decoded_latent.txt into a list of Example objects."""
    examples = []
    current = None

    with open(filepath, "r") as f:
        lines = f.read().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # --- New question ---
        if line.startswith("Question") and "..." in line:
            if current is not None:
                examples.append(current)
            m = re.match(r"Question(\d+)\.\.\.", line)
            qid = int(m.group(1)) if m else len(examples)
            current = Example(question_id=qid, question="", cot="", answer=0.0)
            i += 1
            continue

        # --- Question text ---
        if (current is not None
                and not current.question
                and line
                and not line.startswith("CoT=")
                and not line.startswith("decoded")):
            current.question = line.rstrip(".")
            i += 1
            continue

        # --- CoT + answer ---
        if line.startswith("CoT="):
            m = re.match(r"CoT=(.+),\s*Answer=(.+)", line)
            if m and current:
                current.cot = m.group(1).strip()
                current.answer = float(m.group(2).strip())
            i += 1
            continue

        # --- Latent attended tokens ---
        if line.startswith("decoded") and "latent's attended tokens" in line:
            m = re.match(
                r"decoded (\d+)(?:th|st|nd|rd) latent's attended tokens \(top\d+\): \[(.+)\]",
                line,
            )
            if m and current:
                idx = int(m.group(1))
                attended = re.findall(r"'([^']*)'", m.group(2))

                # Next line: decoded tokens
                i += 1
                decoded = []
                if i < len(lines):
                    nxt = lines[i].strip()
                    m2 = re.match(
                        r"decoded \d+(?:th|st|nd|rd) latent \(top\d+\): \[(.+)\]", nxt
                    )
                    if m2:
                        decoded = re.findall(r"'([^']*)'", m2.group(1))
                    else:
                        i -= 1  # back up

                current.latent_steps.append(
                    LatentStep(index=idx, attended_tokens=attended, decoded_tokens=decoded)
                )
            i += 1
            continue

        # --- "before answer" line (skip) ---
        if line.startswith("decoded before answer"):
            i += 1
            continue

        # --- Model prediction ---
        if line.startswith("Model Prediction:"):
            if current:
                pred = line.replace("Model Prediction:", "").strip()
                current.prediction = pred
                nums = re.findall(r"-?\d+\.?\d*", pred)
                if nums:
                    current.predicted_answer = float(nums[-1])
                    current.correct = abs(current.predicted_answer - current.answer) < 1e-4
            i += 1
            continue

        i += 1

    if current is not None:
        examples.append(current)

    return examples


# ---------------------------------------------------------------------------
# CoT helpers
# ---------------------------------------------------------------------------
def parse_cot_steps(cot: str) -> List[float]:
    """Return the result of each CoT step.

    CoT looks like: <<16-3-4=9>> <<9*2=18>>
    We extract the number after '=' in each <<...>> block.
    """
    results = []
    for block in re.findall(r"<<(.+?)>>", cot):
        parts = block.split("=")
        if len(parts) >= 2:
            try:
                results.append(float(parts[-1]))
            except ValueError:
                pass
    return results


def classify_token(tok: str) -> str:
    """Classify a token as 'number', 'operator', or 'other'."""
    tok = tok.strip()
    if not tok:
        return "other"
    # Operators / punctuation
    if tok in {"+", "-", "*", "/", "=", "<<", ">>", ">", "<", "(", ")", ":", ";",
               ",", ".", "!", "?", "'", '"', "[", "]", "{", "}", "|", "\\", "~",
               "#", "%", "&", "^", "`", ">>", "<<", "->", "=>", "+=", "-=",
               "*=", "/=", "**", "//", "<=", ">=", "!=", "==", "<|endoftext|>",
               "NEWS", "st", "nd", "The", ">>=", "<<="}:
        return "operator"
    # Check if it looks numeric (possibly with leading space)
    cleaned = tok.strip()
    # Pure digits, possibly with decimal point
    if re.match(r"^-?\d+\.?\d*$", cleaned):
        return "number"
    # Contains a digit and short (likely a number token)
    if re.search(r"\d", cleaned) and len(cleaned) <= 6:
        return "number"
    return "other"


def top1(step: LatentStep) -> str:
    """Get the top-1 decoded token, stripped."""
    if step.decoded_tokens:
        return step.decoded_tokens[0].strip()
    return ""


def number_value(tok: str) -> Optional[float]:
    """Try to parse a token as a number."""
    tok = tok.strip()
    try:
        return float(tok)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def compute_token_type_distribution(examples: List[Example]):
    """Chart 1: token-type fraction per latent index."""
    # For each latent 0..7, count types of top-1 decoded token
    counts = {i: Counter() for i in range(8)}  # type -> count
    totals = {i: 0 for i in range(8)}

    for ex in examples:
        for step in ex.latent_steps:
            if step.index > 7:
                continue
            t = classify_token(top1(step))
            counts[step.index][t] += 1
            totals[step.index] += 1

    labels = [f"z{i}" for i in range(8)]
    fracs = {"number": [], "operator": [], "other": []}
    for i in range(8):
        tot = max(totals[i], 1)
        fracs["number"].append(counts[i]["number"] / tot)
        fracs["operator"].append(counts[i]["operator"] / tot)
        fracs["other"].append(counts[i]["other"] / tot)

    return labels, fracs, counts, totals


def compute_cot_tracking(examples: List[Example]):
    """Chart 2: accuracy of even-latent top-1 matching the CoT step result."""
    # Even latents z0,z2,z4,z6  ->  CoT steps 0,1,2,3
    even_indices = [0, 2, 4, 6]
    step_map = {0: 0, 2: 1, 4: 2, 6: 3}  # latent_idx -> cot_step_idx

    match_counts = {i: 0 for i in even_indices}
    attempt_counts = {i: 0 for i in even_indices}

    for ex in examples:
        cot_results = parse_cot_steps(ex.cot)
        step_lookup = {s.index: s for s in ex.latent_steps}
        for li in even_indices:
            ci = step_map[li]
            if ci >= len(cot_results):
                continue
            if li not in step_lookup:
                continue
            attempt_counts[li] += 1
            decoded_val = number_value(top1(step_lookup[li]))
            if decoded_val is not None and abs(decoded_val - cot_results[ci]) < 1e-4:
                match_counts[li] += 1

    accuracies = {}
    for li in even_indices:
        if attempt_counts[li] > 0:
            accuracies[li] = match_counts[li] / attempt_counts[li]
        else:
            accuracies[li] = 0.0

    return even_indices, accuracies, match_counts, attempt_counts


def compute_convergence(examples: List[Example]):
    """Chart 3: repetition / early convergence of even latents."""
    # Pairwise: z0==z2, z2==z4, z4==z6
    # Full: z2==z4==z6
    # Broken down by number of CoT steps

    pairs = [(0, 2), (2, 4), (4, 6)]
    # group by num_steps
    step_groups = {}  # n_steps -> {pair: (match, total), 'full': (match, total)}

    for ex in examples:
        n_steps = len(parse_cot_steps(ex.cot))
        if n_steps < 1:
            continue
        grp_key = min(n_steps, 4)  # 1,2,3,4+
        if grp_key not in step_groups:
            step_groups[grp_key] = {p: [0, 0] for p in pairs}
            step_groups[grp_key]["full"] = [0, 0]

        step_lookup = {s.index: top1(s) for s in ex.latent_steps}

        for (a, b) in pairs:
            if a in step_lookup and b in step_lookup:
                step_groups[grp_key][(a, b)][1] += 1
                if step_lookup[a] == step_lookup[b]:
                    step_groups[grp_key][(a, b)][0] += 1

        # Full convergence: z2==z4==z6
        if all(k in step_lookup for k in [2, 4, 6]):
            step_groups[grp_key]["full"][1] += 1
            if step_lookup[2] == step_lookup[4] == step_lookup[6]:
                step_groups[grp_key]["full"][0] += 1

    return pairs, step_groups


def compute_final_step_location(examples: List[Example]):
    """Chart 4: where is the final CoT step computed?"""
    categories = Counter()  # "In latents", "In answer generation", "Unclear"

    for ex in examples:
        cot_results = parse_cot_steps(ex.cot)
        if len(cot_results) < 2:
            continue  # skip single-step (no "final step" ambiguity)

        final_result = cot_results[-1]
        step_lookup = {s.index: s for s in ex.latent_steps}

        # Check if final result appears in any even latent's top-1
        found_in_latent = False
        for li in [0, 2, 4, 6]:
            if li in step_lookup:
                v = number_value(top1(step_lookup[li]))
                if v is not None and abs(v - final_result) < 1e-4:
                    found_in_latent = True
                    break

        # Check if model prediction matches the ground truth answer
        pred_matches = (ex.predicted_answer is not None
                        and abs(ex.predicted_answer - ex.answer) < 1e-4)

        if found_in_latent:
            categories["Final step in latents"] += 1
        elif pred_matches:
            categories["Final step in answer generation"] += 1
        else:
            categories["Neither / unclear"] += 1

    return categories


def compute_accuracy_by_complexity(examples: List[Example]):
    """Chart 5: accuracy by number of CoT steps."""
    groups = {}  # n_steps -> (correct, total)
    for ex in examples:
        n = len(parse_cot_steps(ex.cot))
        n = max(n, 1)
        key = min(n, 5)  # 1,2,3,4,5+
        if key not in groups:
            groups[key] = [0, 0]
        groups[key][1] += 1
        if ex.correct:
            groups[key][0] += 1
    return groups


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_chart1(labels, fracs, counts, totals, outdir):
    """Stacked bar: token type distribution per latent."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    w = 0.55

    bottom = np.zeros(len(labels))
    num_vals = np.array(fracs["number"])
    op_vals = np.array(fracs["operator"])
    oth_vals = np.array(fracs["other"])

    ax.bar(x, num_vals, w, label="Number", color=C_NUM, bottom=bottom)
    bottom += num_vals
    ax.bar(x, op_vals, w, label="Operator / punct.", color=C_OP, bottom=bottom)
    bottom += op_vals
    ax.bar(x, oth_vals, w, label="Other / word", color=C_OTHER, bottom=bottom)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of top-1 decoded tokens")
    ax.set_xlabel("Latent index")
    ax.set_title("Token Type of Top-1 Decoded Latent")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.9)

    # Annotate percentages on bars for even/odd
    for i in range(8):
        dominant = num_vals[i] if i % 2 == 0 else op_vals[i]
        pct = f"{dominant*100:.1f}%"
        ax.text(x[i], 0.5, pct, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "chart1_token_type_dist.pdf"))
    fig.savefig(os.path.join(outdir, "chart1_token_type_dist.png"))
    plt.close(fig)
    return fig


def plot_chart2(even_indices, accuracies, match_counts, attempt_counts, outdir):
    """Bar chart: CoT step tracking accuracy per even latent."""
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [f"z{i}" for i in even_indices]
    accs = [accuracies[i] for i in even_indices]
    x = np.arange(len(labels))

    bars = ax.bar(x, accs, 0.5, color=[PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction matching CoT step result")
    ax.set_xlabel("Even latent index")
    ax.set_title("CoT Step Tracking Accuracy")
    ax.set_ylim(0, 1.05)

    for i, (bar, li) in enumerate(zip(bars, even_indices)):
        n = attempt_counts[li]
        m = match_counts[li]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m}/{n}\n({accs[i]*100:.1f}%)",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "chart2_cot_tracking.pdf"))
    fig.savefig(os.path.join(outdir, "chart2_cot_tracking.png"))
    plt.close(fig)
    return fig


def plot_chart3(pairs, step_groups, outdir):
    """Grouped bar chart: repetition rates by complexity."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sorted_keys = sorted(step_groups.keys())
    group_labels = []
    for k in sorted_keys:
        group_labels.append(f"{k}-step" if k < 4 else "4+-step")

    pair_labels = [f"z{a}=z{b}" for (a, b) in pairs] + ["z2=z4=z6"]
    n_bars = len(pair_labels)
    x = np.arange(len(sorted_keys))
    w = 0.18

    for j, pl in enumerate(pair_labels):
        rates = []
        for k in sorted_keys:
            grp = step_groups[k]
            if pl == "z2=z4=z6":
                m, t = grp["full"]
            else:
                pair = pairs[j]
                m, t = grp.get(pair, (0, 1))
            rates.append(m / max(t, 1))
        offset = (j - n_bars / 2 + 0.5) * w
        bars = ax.bar(x + offset, rates, w, label=pl, color=PALETTE[j])
        for bi, bar in enumerate(bars):
            if rates[bi] > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{rates[bi]*100:.0f}%", ha="center", va="bottom",
                        fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Repetition rate")
    ax.set_xlabel("Problem complexity (CoT steps)")
    ax.set_title("Early Convergence / Repetition of Even Latents")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "chart3_convergence.pdf"))
    fig.savefig(os.path.join(outdir, "chart3_convergence.png"))
    plt.close(fig)
    return fig


def plot_chart4(categories, outdir):
    """Pie chart: where is the final CoT step computed."""
    fig, ax = plt.subplots(figsize=(5, 4))

    labels = list(categories.keys())
    sizes = list(categories.values())
    colors = [PALETTE[0], PALETTE[1], PALETTE[4]][:len(labels)]
    explode = [0.03] * len(labels)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=140,
        colors=colors, explode=explode, textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    ax.set_title("Where Is the Final CoT Step Computed?\n(multi-step problems only)")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "chart4_final_step_location.pdf"))
    fig.savefig(os.path.join(outdir, "chart4_final_step_location.png"))
    plt.close(fig)
    return fig


def plot_chart5(groups, outdir):
    """Bar chart: accuracy by problem complexity."""
    fig, ax1 = plt.subplots(figsize=(5.5, 4))

    sorted_keys = sorted(groups.keys())
    labels = [f"{k}-step" if k < 5 else "5+-step" for k in sorted_keys]
    accs = [groups[k][0] / max(groups[k][1], 1) for k in sorted_keys]
    counts = [groups[k][1] for k in sorted_keys]
    x = np.arange(len(labels))

    bars = ax1.bar(x, accs, 0.5, color=PALETTE[0], alpha=0.85, label="Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Accuracy (correct predictions)")
    ax1.set_xlabel("Problem complexity (CoT steps)")
    ax1.set_title("Model Accuracy by Problem Complexity")
    ax1.set_ylim(0, 1.15)

    # Annotate with count
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{accs[i]*100:.1f}%\n(n={counts[i]})",
                 ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "chart5_accuracy_by_complexity.pdf"))
    fig.savefig(os.path.join(outdir, "chart5_accuracy_by_complexity.png"))
    plt.close(fig)
    return fig


def plot_combined(examples, outdir,
                  data1, data2, data3, data4, data5):
    """Combined figure with all 5 charts as subplots."""
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.30)

    # --- Chart 1: Token type ---
    ax1 = fig.add_subplot(gs[0, 0])
    labels, fracs, _, _ = data1
    x = np.arange(len(labels))
    w = 0.55
    bottom = np.zeros(len(labels))
    num_vals = np.array(fracs["number"])
    op_vals = np.array(fracs["operator"])
    oth_vals = np.array(fracs["other"])
    ax1.bar(x, num_vals, w, label="Number", color=C_NUM, bottom=bottom)
    bottom += num_vals
    ax1.bar(x, op_vals, w, label="Operator", color=C_OP, bottom=bottom)
    bottom += op_vals
    ax1.bar(x, oth_vals, w, label="Other", color=C_OTHER, bottom=bottom)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Fraction"); ax1.set_xlabel("Latent index")
    ax1.set_title("(a) Token Type Distribution"); ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=7, loc="upper right")

    # --- Chart 2: CoT tracking ---
    ax2 = fig.add_subplot(gs[0, 1])
    even_idx, accs_dict, mc, ac = data2
    acc_vals = [accs_dict[i] for i in even_idx]
    xlabels = [f"z{i}" for i in even_idx]
    ax2.bar(np.arange(len(xlabels)), acc_vals, 0.5,
            color=[PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]])
    ax2.set_xticks(np.arange(len(xlabels))); ax2.set_xticklabels(xlabels)
    ax2.set_ylabel("Accuracy"); ax2.set_xlabel("Even latent")
    ax2.set_title("(b) CoT Step Tracking"); ax2.set_ylim(0, 1.05)
    for i, li in enumerate(even_idx):
        ax2.text(i, acc_vals[i] + 0.02, f"{acc_vals[i]*100:.1f}%",
                 ha="center", fontsize=7)

    # --- Chart 3: Convergence ---
    ax3 = fig.add_subplot(gs[1, 0])
    pairs, step_groups = data3
    sorted_keys = sorted(step_groups.keys())
    grp_labels = [f"{k}-step" if k < 4 else "4+" for k in sorted_keys]
    pair_labels = [f"z{a}=z{b}" for (a, b) in pairs] + ["z2=z4=z6"]
    n_bars = len(pair_labels)
    xc = np.arange(len(sorted_keys))
    wc = 0.18
    for j, pl in enumerate(pair_labels):
        rates = []
        for k in sorted_keys:
            grp = step_groups[k]
            if pl == "z2=z4=z6":
                m, t = grp["full"]
            else:
                m, t = grp.get(pairs[j], (0, 1))
            rates.append(m / max(t, 1))
        ax3.bar(xc + (j - n_bars / 2 + 0.5) * wc, rates, wc,
                label=pl, color=PALETTE[j])
    ax3.set_xticks(xc); ax3.set_xticklabels(grp_labels)
    ax3.set_ylabel("Repetition rate"); ax3.set_xlabel("CoT steps")
    ax3.set_title("(c) Early Convergence"); ax3.set_ylim(0, 1.15)
    ax3.legend(fontsize=6, loc="upper right")

    # --- Chart 4: Final step location ---
    ax4 = fig.add_subplot(gs[1, 1])
    cats = data4
    cat_labels = list(cats.keys())
    cat_sizes = list(cats.values())
    cat_colors = [PALETTE[0], PALETTE[1], PALETTE[4]][:len(cat_labels)]
    ax4.pie(cat_sizes, labels=cat_labels, autopct="%1.1f%%", startangle=140,
            colors=cat_colors, textprops={"fontsize": 8})
    ax4.set_title("(d) Final Step Location")

    # --- Chart 5: Accuracy by complexity ---
    ax5 = fig.add_subplot(gs[2, 0])
    groups = data5
    sk5 = sorted(groups.keys())
    lb5 = [f"{k}" if k < 5 else "5+" for k in sk5]
    ac5 = [groups[k][0] / max(groups[k][1], 1) for k in sk5]
    cn5 = [groups[k][1] for k in sk5]
    bars5 = ax5.bar(np.arange(len(lb5)), ac5, 0.5, color=PALETTE[0], alpha=0.85)
    ax5.set_xticks(np.arange(len(lb5))); ax5.set_xticklabels(lb5)
    ax5.set_ylabel("Accuracy"); ax5.set_xlabel("CoT steps")
    ax5.set_title("(e) Accuracy by Complexity"); ax5.set_ylim(0, 1.15)
    for i, bar in enumerate(bars5):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{ac5[i]*100:.1f}%\nn={cn5[i]}", ha="center", fontsize=7)

    fig.savefig(os.path.join(outdir, "combined_all_charts.pdf"))
    fig.savefig(os.path.join(outdir, "combined_all_charts.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
def print_summary(examples, data1, data2, data3, data4, data5):
    n = len(examples)
    correct = sum(1 for e in examples if e.correct)
    print("=" * 70)
    print(f"CODI Latent Analysis  --  {n} examples, {correct}/{n} correct ({100*correct/n:.1f}%)")
    print("=" * 70)

    # Chart 1
    labels, fracs, counts, totals = data1
    print("\n--- Chart 1: Token Type Distribution (top-1 decoded) ---")
    for i in range(8):
        tag = "even" if i % 2 == 0 else "odd"
        print(f"  z{i} ({tag}):  number={fracs['number'][i]*100:5.1f}%  "
              f"operator={fracs['operator'][i]*100:5.1f}%  "
              f"other={fracs['other'][i]*100:5.1f}%  (n={totals[i]})")

    even_num = np.mean([fracs["number"][i] for i in [0, 2, 4, 6]])
    odd_op = np.mean([fracs["operator"][i] for i in [1, 3, 5]])
    print(f"  => Even latents predict numbers {even_num*100:.1f}% of the time")
    print(f"  => Odd latents predict operators {odd_op*100:.1f}% of the time")

    # Chart 2
    even_idx, accs, mc, ac = data2
    print("\n--- Chart 2: CoT Step Tracking Accuracy ---")
    for li in even_idx:
        print(f"  z{li} -> CoT step {li//2}: {mc[li]}/{ac[li]} = {accs[li]*100:.1f}%")

    # Chart 3
    pairs, step_groups = data3
    print("\n--- Chart 3: Early Convergence ---")
    for k in sorted(step_groups.keys()):
        lbl = f"{k}-step" if k < 4 else "4+-step"
        grp = step_groups[k]
        parts = []
        for (a, b) in pairs:
            m, t = grp[(a, b)]
            parts.append(f"z{a}=z{b}: {m}/{t} ({100*m/max(t,1):.0f}%)")
        m, t = grp["full"]
        parts.append(f"z2=z4=z6: {m}/{t} ({100*m/max(t,1):.0f}%)")
        print(f"  {lbl}: {', '.join(parts)}")

    # Chart 4
    cats = data4
    total_multi = sum(cats.values())
    print("\n--- Chart 4: Final Step Location (multi-step only) ---")
    for k, v in cats.items():
        print(f"  {k}: {v}/{total_multi} ({100*v/max(total_multi,1):.1f}%)")

    # Chart 5
    groups = data5
    print("\n--- Chart 5: Accuracy by Complexity ---")
    for k in sorted(groups.keys()):
        c, t = groups[k]
        lbl = f"{k}-step" if k < 5 else "5+-step"
        print(f"  {lbl}: {c}/{t} = {100*c/max(t,1):.1f}%")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze CODI latent predictions")
    parser.add_argument("--input", "-i", default="outputs/decoded_latent.txt")
    parser.add_argument("--outdir", "-o", default="outputs/charts")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Parsing {args.input} ...")
    examples = parse_file(args.input)
    print(f"Loaded {len(examples)} examples.\n")

    # Compute all analyses
    data1 = compute_token_type_distribution(examples)
    data2 = compute_cot_tracking(examples)
    data3 = compute_convergence(examples)
    data4 = compute_final_step_location(examples)
    data5 = compute_accuracy_by_complexity(examples)

    # Plot individual charts
    plot_chart1(*data1, args.outdir)
    print("  Saved chart 1: token type distribution")
    plot_chart2(*data2, args.outdir)
    print("  Saved chart 2: CoT step tracking accuracy")
    plot_chart3(*data3, args.outdir)
    print("  Saved chart 3: early convergence")
    plot_chart4(data4, args.outdir)
    print("  Saved chart 4: final step location")
    plot_chart5(data5, args.outdir)
    print("  Saved chart 5: accuracy by complexity")

    # Combined figure
    plot_combined(examples, args.outdir, data1, data2, data3, data4, data5)
    print("  Saved combined figure")

    # Print text summary
    print()
    print_summary(examples, data1, data2, data3, data4, data5)


if __name__ == "__main__":
    main()
