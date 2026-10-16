#!/bin/bash
# Slurm reads lines starting with #SBATCH (do not remove or change to ##SBATCH).
#SBATCH --job-name=genagents-wavelength-gemma-3-12b-it
#SBATCH --account=bdks-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus=1
#SBATCH --mem=300g
#SBATCH --time=25:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yueshen7@illinois.edu
#SBATCH --output=agent_bank/scripts-agent-filtering/wavelength_logs/slurm-%j.out
#SBATCH --error=agent_bank/scripts-agent-filtering/wavelength_logs/slurm-%j.err

set -euo pipefail

REPO_ROOT="/projects/bdks/yueshen7/repos/genagents"
LOG_DIR="$REPO_ROOT/agent_bank/scripts-agent-filtering/wavelength_logs"
TARGET_ATTEMPTS="${TARGET_ATTEMPTS:-3}"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
source venv/bin/activate

echo "========== Wavelength patch started $(date -Iseconds) =========="
echo "Population: gemma-3-12b-it"
echo "Target attempts: $TARGET_ATTEMPTS"

CUDA_VISIBLE_DEVICES=0 python3 -u Wavelength_Game/run_wavelength_patch_attempts.py \
  --base agent_bank/populations/gemma-3-12b-it_agents \
  --target-attempts "$TARGET_ATTEMPTS" \
  --model-choice gemma-3-12b-it \
  >> "$LOG_DIR/wavelength_patch_gemma-3-12b-it_${SLURM_JOB_ID}.log" 2>&1

echo "========== Wavelength patch finished $(date -Iseconds) =========="

