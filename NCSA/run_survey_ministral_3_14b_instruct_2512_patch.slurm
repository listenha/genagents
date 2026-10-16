#!/bin/bash
# Slurm reads lines starting with #SBATCH (do not remove or change to ##SBATCH).
#SBATCH --job-name=genagents-survey-ministral-3-14b
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
#SBATCH --output=agent_bank/scripts-agent-filtering/survey_logs/slurm-%j.out
#SBATCH --error=agent_bank/scripts-agent-filtering/survey_logs/slurm-%j.err

set -euo pipefail

REPO_ROOT="/projects/bdks/yueshen7/repos/genagents"
LOG_DIR="$REPO_ROOT/agent_bank/scripts-agent-filtering/survey_logs"
TARGET_ATTEMPTS="${TARGET_ATTEMPTS:-3}"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
source venv/bin/activate

echo "========== Survey patch started $(date -Iseconds) =========="
echo "Population: Ministral-3-14B-Instruct-2512"
echo "Target attempts: $TARGET_ATTEMPTS"

CUDA_VISIBLE_DEVICES=0 python3 -u Surveys/run_survey_patch_attempts.py \
  --base agent_bank/populations/Ministral-3-14B-Instruct-2512_agents \
  --target-attempts "$TARGET_ATTEMPTS" \
  --model-choice ministral-3-14b-instruct-2512 \
  >> "$LOG_DIR/survey_patch_ministral-3-14b-instruct-2512_${SLURM_JOB_ID}.log" 2>&1

echo "========== Survey patch finished $(date -Iseconds) =========="

