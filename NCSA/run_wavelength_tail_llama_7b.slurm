#!/bin/bash
#SBATCH --job-name=genagents-wl-tail-llama-7b
#SBATCH --account=bdks-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus=1
#SBATCH --mem=100g
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yueshen7@illinois.edu
#SBATCH --output=agent_bank/scripts-agent-filtering/wavelength_logs/slurm-wl-tail-llama-7b-%j.out
#SBATCH --error=agent_bank/scripts-agent-filtering/wavelength_logs/slurm-wl-tail-llama-7b-%j.err

set -e
REPO_ROOT="/projects/bdks/yueshen7/repos/genagents"
cd "$REPO_ROOT"
source venv/bin/activate
export TARGET_ATTEMPTS="${TARGET_ATTEMPTS:-3}"
LOG_DIR="$REPO_ROOT/agent_bank/scripts-agent-filtering/wavelength_logs"
mkdir -p "$LOG_DIR"

echo "========== Wavelength tail (Llama + 7b) started $(date -Iseconds) =========="
CUDA_VISIBLE_DEVICES=0 python3 -u Wavelength_Game/run_wavelength_patch_attempts.py --base agent_bank/populations/Llama-3.1-8B_agents --target-attempts "$TARGET_ATTEMPTS" --model-choice llama3.1-8b >> "$LOG_DIR/wavelength_tail_llama_$SLURM_JOB_ID.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python3 -u Wavelength_Game/run_wavelength_patch_attempts.py --base agent_bank/populations/Qwen2.5-7B_agents --target-attempts "$TARGET_ATTEMPTS" --model-choice 7b >> "$LOG_DIR/wavelength_tail_7b_$SLURM_JOB_ID.log" 2>&1
echo "========== Wavelength tail (Llama + 7b) finished $(date -Iseconds) =========="
