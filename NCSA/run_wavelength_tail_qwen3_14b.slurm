#!/bin/bash
#SBATCH --job-name=genagents-wl-tail-qwen14b
#SBATCH --account=bdks-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus=1
#SBATCH --mem=100g
#SBATCH --time=05:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yueshen7@illinois.edu
#SBATCH --output=agent_bank/scripts-agent-filtering/wavelength_logs/slurm-wl-tail-qwen14b-%j.out
#SBATCH --error=agent_bank/scripts-agent-filtering/wavelength_logs/slurm-wl-tail-qwen14b-%j.err

set -e
REPO_ROOT="/projects/bdks/yueshen7/repos/genagents"
cd "$REPO_ROOT"
source venv/bin/activate
export TARGET_ATTEMPTS="${TARGET_ATTEMPTS:-3}"
LOG_DIR="$REPO_ROOT/agent_bank/scripts-agent-filtering/wavelength_logs"
mkdir -p "$LOG_DIR"

echo "========== Wavelength tail (Qwen3-14B) started $(date -Iseconds) =========="
CUDA_VISIBLE_DEVICES=0 python3 -u Wavelength_Game/run_wavelength_patch_attempts.py --base agent_bank/populations/Qwen3-14B_agents --target-attempts "$TARGET_ATTEMPTS" --model-choice 14b >> "$LOG_DIR/wavelength_tail_qwen14b_$SLURM_JOB_ID.log" 2>&1
echo "========== Wavelength tail (Qwen3-14B) finished $(date -Iseconds) =========="
