# CODI Codebase Notes

## Overview
CODI = "Compressing Chain-of-Thought into Continuous Space via Self-Distillation"
- Paper: https://arxiv.org/abs/2502.21074
- Accepted EMNLP 2025

## Key Concept
- Compresses chain-of-thought reasoning into **latent tokens** (continuous embeddings)
- Uses self-distillation: teacher (full CoT) -> student (latent tokens)
- Evaluates on GSM8K math reasoning dataset

## Codebase Structure
```
codi/
├── src/model.py      # CODI model class, training args
├── train.py          # Training script
├── test.py           # Evaluation script
├── probe_latent_token.py  # Probe/visualize latent thoughts + attention
├── scripts/          # Shell scripts for running experiments
└── outputs/          # Output files (decoded_latent.txt)
```

## Key Files

### src/model.py
- `CODI` class: wraps a base LM (GPT-2 or LLaMA) with:
  - LoRA adapters (optional)
  - Projection layer (`prj`) for latent embeddings
  - Special tokens: `bot_id` (begin-of-thought), `eot_id` (end-of-thought)
- Loss: CE loss (student) + distillation loss (MSE/SmoothL1) + teacher CE loss
- Uses `attn_implementation="eager"` for attention visualization compatibility

### probe_latent_token.py
- Probes what the model "thinks" at each latent step
- Decodes top-k tokens from latent embeddings via `lm_head`
- **Attention visualization**: shows which input tokens each latent attends to
- `make_eager_attention_mask()`: fixes left-padding + eager attention NaN issue
- Outputs to `outputs/decoded_latent.txt`

## Key Arguments
- `num_latent`: number of latent tokens during training
- `inf_latent_iterations`: number of latent iterations at inference
- `use_prj`: use projection layer for latents
- `prj_dim`: projection hidden dimension

## Pretrained Models
Available on HuggingFace (zen-E):
- zen-E/CODI-gpt2
- zen-E/CODI-llama3.2-1b-Instruct

## Issues Found & Fixed

### 1. Checkpoint path invalid
- Original `scripts/probe_latent_token.sh` referenced `/scratch/prj/...`
- **Fixed**: Download from HuggingFace to `checkpoints/codi-gpt2`

### 2. Eager attention + left padding = NaN (MAJOR)
- **Issue**: With `attn_implementation="eager"` and left-padded batches, padding tokens have fully masked attention rows (all -inf), causing NaN after softmax. NaN propagates through KV-cache.
- **Root cause**: SDPA has `_unmask_unattended()` fix, eager doesn't apply it automatically
- **Symptom**: Accuracy drops from 43% to 12% with batched inference
- **Fix**: `make_eager_attention_mask()` uses `AttentionMaskConverter._unmask_unattended()`
- Reference: https://github.com/huggingface/transformers/issues/35270

## Environment
- Using uv (not conda)
- Python 3.12 via system
- Dependencies installed to `.venv/`

## Experiment Results

### Attention Implementation Comparison
| Configuration | Accuracy |
|--------------|----------|
| SDPA, batch=128 | 43.59% |
| Eager, batch=128, no mask fix | 12.13% |
| Eager, batch=1 (no padding) | 41.55% |
| Eager, batch=128, with mask fix | **41.24%** |

### Attention Visualization Findings
The latent tokens encode intermediate calculations:
- "16 eggs - 3 breakfast" → latent predicts "13"
- "2 bolts / 2" → latent predicts "1"
- "3 sprints × 3 times" → latent predicts "9"

Latents attend to:
- Relevant numbers in the input
- Previous latent tokens (`<LAT0>`, `<LAT1>`, etc.)
- Beginning/end markers (`<BOT>`)

---

## Tier 1 Latent Analysis (2026-02-05)

### Scripts Created
All in `claude/scripts/`:

| Script | GPU? | Purpose |
|--------|------|---------|
| `latent_analysis.py` + `.slurm` | Yes | Logit lens (per-layer lm_head) + attention entropy per latent |
| `early_stopping.py` + `.slurm` | Yes | Accuracy with iterations=0..6 |
| `attention_tracking.py` + `.slurm` | Yes | Top-20 attended input token positions per latent |
| `latent_patching.py` + `.slurm` | Yes | Causal interventions: ablation, token forcing, frozen latent |
| `analyze_all.py` | No | Charts for early stopping, logit lens, entropy, trajectory |
| `analyze_causal.py` | No | Charts for attention tracking + patching results |

All SLURM scripts need `PYTHONPATH=$CODI_DIR` since scripts live in `claude/scripts/`.

### Output Files
- `outputs/latent_data.pt` — logit lens + entropy data (N=1319, 7 latent pos, 13 layers)
- `outputs/early_stopping.json` — accuracy per iteration count
- `outputs/attention_tracking.pt` — top-20 attended input positions per latent
- `outputs/latent_patching.json` — all intervention results
- `outputs/charts/` — all generated charts (PDF + PNG)

### Results Summary

#### B2. Early Stopping — accuracy vs iterations
| Iters | Acc   |
|-------|-------|
| 0     | 14.9% |
| 1     | 23.1% |
| 2     | 21.8% |
| 3     | 35.7% |
| 4     | 35.8% |
| 5     | 40.9% |
| 6     | 41.2% |

Monotonic improvement (except slight dip at iter 2). Biggest jumps at 0→1 and 2→3. Not yet saturating.

#### B5. Logit Lens
- CoT numbers only "crystallize" in the last 2-3 layers (L11-L12)
- z0 has highest match rate (0.69 at L12), declining for later latents
- Even/odd alternation in crystallization layer

#### A4. Attention Entropy — **NOT USEFUL**
- Flat ~1.85 nats across all latent positions
- No difference between correct/incorrect predictions
- Dropped from analysis

#### C3. Token Prediction Trajectory — **NOT USEFUL**
- Categories: Immediate 37.6%, Early converge 35.4%, Late converge 20.4%, No converge 6.5%
- Accuracy correlates (100% → 93.8%) but not very informative
- Dropped from analysis

#### Attention-Input Tracking — **USEFUL**
Operand match rates (top-5 attended input tokens):
- z0 → CoT step 0: **48.5%** match
- z2 → CoT step 1: **31.9%**
- z4 → CoT step 2: **19.6%**
- z6 → CoT step 3: **16.1%**

Source operands (from question) in top-5: **42.6%**
Intermediate operands (from prev step): only **5.7%** — these must come from latent positions, not input.

Attention fraction: z0 = 100% to input, z1-z6 ≈ 44-58% to input (rest to latent positions).

Inter-latent attention: **odd latents chain** — z3→z1 (0.084), z5→z3 (0.085). Odd latents read from previous odd latents.

#### Latent Patching — **MOST USEFUL**

**Ablation** (zero each position):
| Position | Acc   | Delta |
|----------|-------|-------|
| z0       | 40.4% | -0.8  |
| z1       | 35.2% | -6.1  |
| z2       | 36.1% | -5.1  |
| z3       | 33.9% | -7.4  |
| z4       | 33.1% | **-8.2** |
| z5       | 41.0% | -0.2  |
| z6       | 41.2% | +0.0  |

**z5 and z6 are dead positions** — ablation has no effect. Middle positions z1-z4 are load-bearing.

**Token forcing** (replace latent embedding with token embedding):
| Position | Force correct | Force wrong (+1) |
|----------|--------------|-----------------|
| z0       | 38.5% (-2.7) | 25.7% (**-15.5**) |
| z2       | 37.1% (-4.1) | 23.7% (**-17.5**) |
| z4       | 37.0% (-4.2) | 28.1% (-13.2) |
| z6       | 41.2% (+0.0) | 41.2% (+0.0) |

- Force correct HURTS slightly — latent embeds richer info than just the number token
- Force wrong DESTROYS accuracy at z0/z2 — causal proof latents carry real computation
- z6 immune to both — embedding is never read (only KV cache matters)

**Frozen z0**: 22.1% — worse than baseline but better than 0-iteration (14.9%).

### Key Insights
1. **Latents are causally active** — wrong numbers at z0/z2 cause 15-17% accuracy drops
2. **Latent representations are richer than tokens** — forcing correct numbers still hurts
3. **z5/z6 are dead** — model doesn't read their embeddings, only their KV cache entries
4. **Odd latents chain** — z3→z1→z0 attention pattern suggests sequential processing
5. **Source operands are attended** — 48% match at z0, but intermediates come from latent positions not input

### Chart Formatting Issues
- `patch_token_forcing.png` has label overlap issues — needs cleanup
- Some charts could be more compact
