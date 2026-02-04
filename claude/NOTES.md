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
├── probe_latent_token.py  # Probe/visualize latent thoughts
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

### probe_latent_token.py
- Probes what the model "thinks" at each latent step
- Decodes top-k tokens from latent embeddings via `lm_head`
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

## Issues Found
1. **Checkpoint path invalid**: `scripts/probe_latent_token.sh` references `/scratch/prj/...` which doesn't exist on this cluster
2. ~~Need to download HF model~~ - RESOLVED: Downloaded zen-E/CODI-gpt2 to `checkpoints/codi-gpt2`

## Environment
- Using uv (not conda)
- Python 3.12 via system (3.14.2 available via mise but 3.12 used for venv)
- Dependencies installed to `.venv/`

## Experiment Results

### 2026-02-04: probe_latent_token (job 187740)
- Model: CODI-gpt2 (HuggingFace zen-E/CODI-gpt2)
- Dataset: GSM8K-Aug (1319 test examples)
- Config: 6 latent iterations, batch_size=128, greedy=True
- **Accuracy: 43.59%**
- Average CoT length: 6.25 tokens
- Output: `outputs/decoded_latent.txt`
