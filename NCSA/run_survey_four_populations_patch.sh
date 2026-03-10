#!/usr/bin/env bash
# Patch survey attempts for the four agent populations to a target number of
# attempts per agent. Cleans dirty attempts (missing/empty responses) then runs
# new attempts until each agent has TARGET_ATTEMPTS. Uses one question per
# inference (no batch). Run from repo root with venv active.
#
# Tunable: export TARGET_ATTEMPTS=3   (default 3)
# Usage: ./NCSA/run_survey_four_populations_patch.sh

set -e
REPO_ROOT="/projects/bdks/yueshen7/repos/genagents"
SETTINGS="${REPO_ROOT}/simulation_engine/settings.py"
LOG_DIR="${REPO_ROOT}/agent_bank/scripts-agent-filtering/survey_logs"
TARGET_ATTEMPTS="${TARGET_ATTEMPTS:-3}"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"
if [ -n "$VIRTUAL_ENV" ]; then
  echo "Using venv: $VIRTUAL_ENV"
else
  echo "Activating venv..."
  source venv/bin/activate
fi

run_patch_for_model() {
  local choice="$1"
  local base_path="$2"
  local log_file="${LOG_DIR}/survey_patch_${choice}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date -Iseconds)] MODEL_CHOICE=$choice -> $base_path (target_attempts=$TARGET_ATTEMPTS, log: $log_file)"
  sed -i "s/^MODEL_CHOICE = .*/MODEL_CHOICE = \"${choice}\"/" "$SETTINGS"
  python3 -u Surveys/run_survey_patch_attempts.py --base "$base_path" --target-attempts "$TARGET_ATTEMPTS" >> "$log_file" 2>&1
  echo "[$(date -Iseconds)] Done $choice exit_code=$?"
}

echo "========== Survey patch (four populations, target_attempts=$TARGET_ATTEMPTS, started $(date -Iseconds)) =========="
run_patch_for_model "mistral-nemo-2407" "agent_bank/populations/Mistral-Nemo_agents" || true
run_patch_for_model "llama3.1-8b"      "agent_bank/populations/Llama-3.1-8B_agents"    || true
run_patch_for_model "7b"               "agent_bank/populations/Qwen2.5-7B_agents"     || true
run_patch_for_model "14b"              "agent_bank/populations/Qwen3-14B_agents"      || true
echo "========== Finished $(date -Iseconds) =========="
