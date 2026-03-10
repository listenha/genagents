#!/bin/bash
# Slurm reads lines starting with #SBATCH (do not remove or change to ##SBATCH).
#SBATCH --job-name=genagents-survey-tail-llama
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
#SBATCH --output=agent_bank/scripts-agent-filtering/survey_logs/slurm-survey-tail-llama-%j.out
#SBATCH --error=agent_bank/scripts-agent-filtering/survey_logs/slurm-survey-tail-llama-%j.err

set -e
REPO_ROOT="/projects/bdks/yueshen7/repos/genagents"
cd "$REPO_ROOT"
source venv/bin/activate

export TARGET_ATTEMPTS="${TARGET_ATTEMPTS:-3}"
LOG_DIR="$REPO_ROOT/agent_bank/scripts-agent-filtering/survey_logs"
mkdir -p "$LOG_DIR"

echo "========== Survey tail (Llama-3.1-8B) started $(date -Iseconds) =========="
echo "TARGET_ATTEMPTS=$TARGET_ATTEMPTS"

CUDA_VISIBLE_DEVICES=0 python3 -u Surveys/run_survey_patch_attempts.py \
  --base agent_bank/populations/Llama-3.1-8B_agents \
  --target-attempts "$TARGET_ATTEMPTS" \
  --model-choice llama3.1-8b \
  >> "$LOG_DIR/survey_tail_llama_$SLURM_JOB_ID.log" 2>&1

echo "========== Survey tail (Llama-3.1-8B) finished $(date -Iseconds) =========="
