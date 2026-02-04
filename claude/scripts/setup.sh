#!/bin/bash
# Setup script for CODI project
set -e

VAST=/workspace-vast/sshlin
CODI_DIR=$VAST/codi

cd $CODI_DIR

# Create venv if not exists
if [ ! -d ".venv" ]; then
    uv venv --python 3.12
fi

source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Create directories
mkdir -p claude/logs
mkdir -p outputs
mkdir -p checkpoints

echo "Setup complete. Run: sbatch claude/scripts/probe_latent.slurm"
