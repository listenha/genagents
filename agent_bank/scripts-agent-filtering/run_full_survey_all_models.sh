#!/usr/bin/env bash
# Run full survey for all agents in each model-specific population folder,
# using the corresponding base model. Designed to run inside tmux so it
# persists if the connection is lost.
# Usage: run from repo root, e.g. ./agent_bank/scripts-agent-filtering/run_full_survey_all_models.sh

set -e
REPO_ROOT="/taiga/common_resources/yueshen7/genagents"
SETTINGS="${REPO_ROOT}/simulation_engine/settings.py"
LOG_DIR="${REPO_ROOT}/agent_bank/scripts-agent-filtering/survey_logs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"
source venv/bin/activate

# MODEL_CHOICE value : base path (relative to repo root)
run_survey_for_model() {
  local choice="$1"
  local base_path="$2"
  local log_file="${LOG_DIR}/survey_${choice}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date -Iseconds)] Setting MODEL_CHOICE=$choice and running survey for $base_path (log: $log_file)"
  sed -i "s/^MODEL_CHOICE = .*/MODEL_CHOICE = \"${choice}\"/" "$SETTINGS"
  python3 -u Surveys/run_survey.py --base "$base_path" --all >> "$log_file" 2>&1
  local status=$?
  echo "[$(date -Iseconds)] Finished $choice exit_code=$status"
  return $status
}

echo "========== Full survey for all models (started $(date -Iseconds)) =========="
echo "Logs directory: $LOG_DIR"

run_survey_for_model "mistral-nemo-2407" "agent_bank/populations/Mistral-Nemo_agents" || true
run_survey_for_model "llama3.1-8b"      "agent_bank/populations/Llama-3.1-8B_agents"    || true
run_survey_for_model "7b"               "agent_bank/populations/Qwen2.5-7B_agents"     || true
run_survey_for_model "14b"              "agent_bank/populations/Qwen3-14B_agents"      || true

# Restore default model choice
sed -i 's/^MODEL_CHOICE = .*/MODEL_CHOICE = "14b"/' "$SETTINGS"
echo "========== All survey runs completed $(date -Iseconds) =========="
