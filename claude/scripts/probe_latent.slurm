#!/bin/bash
#SBATCH --job-name=probe_latent
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=claude/logs/probe_latent_%j.out
#SBATCH --error=claude/logs/probe_latent_%j.err

set -e

VAST=/workspace-vast/sshlin
CODI_DIR=$VAST/codi

# Setup node (HF/wandb auth)
bash $VAST/afel-inst/setup-node.sh

cd $CODI_DIR

# Activate venv
source .venv/bin/activate

# Checkpoint directory (download from HF if needed)
CKPT_DIR=$CODI_DIR/checkpoints/codi-gpt2

if [ ! -d "$CKPT_DIR" ]; then
    echo "Downloading checkpoint from HuggingFace..."
    mkdir -p $CODI_DIR/checkpoints
    huggingface-cli download zen-E/CODI-gpt2 --local-dir $CKPT_DIR
fi

# Run probe
python probe_latent_token.py \
    --data_name "zen-E/GSM8k-Aug" \
    --output_dir "outputs" \
    --model_name_or_path gpt2 \
    --seed 11 \
    --model_max_length 512 \
    --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --batch_size 128 \
    --greedy True \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 768 \
    --prj_no_ln False \
    --prj_dropout 0.0 \
    --inf_latent_iterations 6 \
    --inf_num_iterations 1 \
    --remove_eos True \
    --use_lora True \
    --ckpt_dir $CKPT_DIR
