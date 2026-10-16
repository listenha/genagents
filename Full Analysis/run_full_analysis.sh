#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/projects/bdks/yueshen7/repos/genagents"
OUT_ROOT="$REPO_ROOT/Full Analysis/results"

SURVEY_OUT="$OUT_ROOT/survey_report"
WGAME_OUT="$OUT_ROOT/wgame_report"
CORR_OUT="$OUT_ROOT/correlation-results"
VIZ_OUT="$OUT_ROOT/visualizations"

AGENT_RANGE="0000-0199"
ATTEMPTS="1-3"

cd "$REPO_ROOT"
source venv/bin/activate

mkdir -p "$SURVEY_OUT" "$WGAME_OUT" "$CORR_OUT"
mkdir -p "$VIZ_OUT"

run_survey() {
  local pop_name="$1"
  local pop_path="$2"
  echo "[survey] $pop_name"
  python -u Surveys/analyze_survey.py \
    --base "$pop_path" \
    --range "$AGENT_RANGE" \
    --attempts "$ATTEMPTS" \
    --output "$SURVEY_OUT/survey_${pop_name}.html" \
    --csv "$SURVEY_OUT/survey_${pop_name}_csv"
}

run_wgame() {
  local pop_name="$1"
  local pop_path="$2"
  echo "[wavelength] $pop_name"
  python -u Wavelength_Game/analyze_wavelength.py \
    --base "$pop_path" \
    --range "$AGENT_RANGE" \
    --attempts "$ATTEMPTS" \
    --output "$WGAME_OUT/wavelength_${pop_name}.html" \
    --csv "$WGAME_OUT/wavelength_${pop_name}_csv"
}

run_translate_survey() {
  local pop_name="$1"
  local pop_path="$2"
  echo "[translate survey -> distilled] $pop_name"
  python -u Surveys/translate_responses.py \
    --base "$pop_path" \
    --range "$AGENT_RANGE"
}

run_corr() {
  local pop_name="$1"
  local pop_path="$2"
  echo "[correlation] $pop_name"
  python -u agent_bank/Analysis/analyze_profile_correlation.py \
    --base "$pop_path" \
    --output "$CORR_OUT/$pop_name"
}

run_corr_viz() {
  local pop_name="$1"
  echo "[correlation viz] $pop_name"
  python -u agent_bank/Analysis/visualize_results.py \
    --input-dir "$CORR_OUT/$pop_name" \
    --output-dir "$VIZ_OUT/$pop_name"
}

# pop_name -> population folder
run_all_for_population() {
  local pop_name="$1"
  local pop_dir="$2"
  local pop_path="$REPO_ROOT/$pop_dir"

  run_translate_survey "$pop_name" "$pop_path"
  run_survey "$pop_name" "$pop_path"
  run_wgame "$pop_name" "$pop_path"
  run_corr  "$pop_name" "$pop_path"
  run_corr_viz "$pop_name"
}

echo "========== Full analysis run started $(date -Iseconds) =========="
echo "Output root: $OUT_ROOT"

declare -A POPS
POPS[mistral-nemo-2407]="agent_bank/populations/Mistral-Nemo_agents"
POPS[llama3.1-8b]="agent_bank/populations/Llama-3.1-8B_agents"
POPS[7b]="agent_bank/populations/Qwen2.5-7B_agents"
POPS[14b]="agent_bank/populations/Qwen3-14B_agents"

for pop in mistral-nemo-2407 llama3.1-8b 7b 14b; do
  run_all_for_population "$pop" "${POPS[$pop]}"
  echo ""
done

echo "========== Full analysis run finished $(date -Iseconds) =========="
