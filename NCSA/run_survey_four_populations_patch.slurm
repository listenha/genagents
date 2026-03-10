#!/bin/bash
# Slurm reads lines starting with #SBATCH (do not remove or change to ##SBATCH).
#SBATCH --job-name=genagents-survey-patch
#SBATCH --account=bdks-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus=4
#SBATCH --mem=300g
#SBATCH --time=20:00:00
# Email when job ends or fails (replace YOUR_NETID@illinois.edu with your email)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yueshen7@illinois.edu
#SBATCH --output=agent_bank/scripts-agent-filtering/survey_logs/slurm-%j.out
#SBATCH --error=agent_bank/scripts-agent-filtering/survey_logs/slurm-%j.err

# Run survey patch on 4 GPUs in parallel (one population per GPU).
# Submit from repo root: sbatch NCSA/run_survey_four_populations_patch.slurm
# The job runs on a compute node when allocated; no need for tmux—it persists after you disconnect.

set -e
REPO_ROOT="/projects/bdks/yueshen7/repos/genagents"
cd "$REPO_ROOT"
source venv/bin/activate
export TARGET_ATTEMPTS="${TARGET_ATTEMPTS:-3}"
LOG_DIR="$REPO_ROOT/agent_bank/scripts-agent-filtering/survey_logs"
mkdir -p "$LOG_DIR"

echo "========== Survey patch job started $(date -Iseconds) =========="
echo "TARGET_ATTEMPTS=$TARGET_ATTEMPTS"
echo "Running 4 populations in parallel on 4 GPUs"

# Run one population per GPU in parallel (each process uses its own MODEL_CHOICE and GPU)
( CUDA_VISIBLE_DEVICES=0 python3 -u Surveys/run_survey_patch_attempts.py --base agent_bank/populations/Mistral-Nemo_agents --target-attempts "$TARGET_ATTEMPTS" --model-choice mistral-nemo-2407 >> "$LOG_DIR/survey_patch_mistral_$$.log" 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 python3 -u Surveys/run_survey_patch_attempts.py --base agent_bank/populations/Llama-3.1-8B_agents    --target-attempts "$TARGET_ATTEMPTS" --model-choice llama3.1-8b      >> "$LOG_DIR/survey_patch_llama_$$.log" 2>&1 ) &
( CUDA_VISIBLE_DEVICES=2 python3 -u Surveys/run_survey_patch_attempts.py --base agent_bank/populations/Qwen2.5-7B_agents     --target-attempts "$TARGET_ATTEMPTS" --model-choice 7b               >> "$LOG_DIR/survey_patch_7b_$$.log" 2>&1 ) &
( CUDA_VISIBLE_DEVICES=3 python3 -u Surveys/run_survey_patch_attempts.py --base agent_bank/populations/Qwen3-14B_agents      --target-attempts "$TARGET_ATTEMPTS" --model-choice 14b              >> "$LOG_DIR/survey_patch_14b_$$.log" 2>&1 ) &
wait

echo "========== Survey patch job finished $(date -Iseconds) =========="
