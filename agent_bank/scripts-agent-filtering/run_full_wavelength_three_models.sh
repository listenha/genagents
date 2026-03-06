#!/usr/bin/env bash
# Run Wavelength game once for all agents in Mistral-Nemo, Llama-3.1-8B, and Qwen2.5-7B
# population folders using the corresponding base model. Excludes Qwen3-14B.
# Usage: run from repo root inside tmux.

set -e
REPO_ROOT="/taiga/common_resources/yueshen7/genagents"
SETTINGS="${REPO_ROOT}/simulation_engine/settings.py"
LOG_DIR="${REPO_ROOT}/agent_bank/scripts-agent-filtering/wavelength_logs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"
source venv/bin/activate

run_wavelength_for_model() {
  local choice="$1"
  local base_path="$2"
  local log_file="${LOG_DIR}/wavelength_${choice}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date -Iseconds)] Setting MODEL_CHOICE=$choice and running Wavelength for $base_path (log: $log_file)"
  sed -i "s/^MODEL_CHOICE = .*/MODEL_CHOICE = \"${choice}\"/" "$SETTINGS"
  python3 -u Wavelength_Game/run_wavelength.py --base "$base_path" --all >> "$log_file" 2>&1
  local status=$?
  echo "[$(date -Iseconds)] Finished $choice exit_code=$status"
  return $status
}

echo "========== Full Wavelength game for 3 models (started $(date -Iseconds)) =========="
echo "Logs directory: $LOG_DIR"

run_wavelength_for_model "mistral-nemo-2407" "agent_bank/populations/Mistral-Nemo_agents" || true
run_wavelength_for_model "llama3.1-8b"      "agent_bank/populations/Llama-3.1-8B_agents"    || true
run_wavelength_for_model "7b"               "agent_bank/populations/Qwen2.5-7B_agents"     || true

# Restore default model choice
sed -i 's/^MODEL_CHOICE = .*/MODEL_CHOICE = "14b"/' "$SETTINGS"
echo "========== All Wavelength runs completed $(date -Iseconds) =========="
