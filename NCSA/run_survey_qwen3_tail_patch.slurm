#!/bin/bash
# Slurm reads lines starting with #SBATCH (do not remove or change to ##SBATCH).
#SBATCH --job-name=genagents-survey-qwen3-tail
#SBATCH --account=bdks-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus=1
#SBATCH --mem=100g
#SBATCH --time=03:00:00
# Email when job ends or fails (replace with your email)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yueshen7@illinois.edu
#SBATCH --output=agent_bank/scripts-agent-filtering/survey_logs/slurm-qwen3-tail-%j.out
#SBATCH --error=agent_bank/scripts-agent-filtering/survey_logs/slurm-qwen3-tail-%j.err

# Patch Qwen3-14B population to 3 attempts per agent, using a single GPU.
# Safe to run on the whole population; it skips agents already at 3 clean attempts.

set -e
REPO_ROOT="/projects/bdks/yueshen7/repos/genagents"
cd "$REPO_ROOT"
source venv/bin/activate

export TARGET_ATTEMPTS="${TARGET_ATTEMPTS:-3}"
LOG_DIR="$REPO_ROOT/agent_bank/scripts-agent-filtering/survey_logs"
mkdir -p "$LOG_DIR"

echo "========== Qwen3-14B survey tail patch started $(date -Iseconds) =========="
echo "TARGET_ATTEMPTS=$TARGET_ATTEMPTS"

echo "Using 1 GPU; Qwen3-14B population only"
CUDA_VISIBLE_DEVICES=0 python3 -u Surveys/run_survey_patch_attempts.py \
  --base agent_bank/populations/Qwen3-14B_agents \
  --target-attempts "$TARGET_ATTEMPTS" \
  --model-choice 14b \
  >> "$LOG_DIR/survey_patch_qwen3_tail_$SLURM_JOB_ID.log" 2>&1

echo "========== Qwen3-14B survey tail patch finished $(date -Iseconds) =========="
