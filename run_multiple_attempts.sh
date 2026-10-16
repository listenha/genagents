#!/usr/bin/env bash
# Run the survey or the Wavelength game NUM_ATTEMPTS times over a set of agents.
# Each run appends one attempt per agent. To end up with exactly N clean
# attempts per agent, prefer Surveys/run_survey_patch_attempts.py or
# Wavelength_Game/run_wavelength_patch_attempts.py.
#
# Usage:  ./run_multiple_attempts.sh survey|wavelength
# Every setting below can be overridden from the environment, e.g.
#   GENAGENTS_MODEL=7b BASE_PATH=agent_bank/populations/Qwen2.5-7B_agents \
#   AGENT_RANGE=0000-0049 ./run_multiple_attempts.sh wavelength

set -u

TASK="${1:-}"
case "${TASK}" in
  survey)     RUNNER="Surveys/run_survey.py";              BATCH_FLAG="--batch-by-section" ;;
  wavelength) RUNNER="Wavelength_Game/run_wavelength.py";  BATCH_FLAG="--batch-by-header" ;;
  *) echo "Usage: $0 survey|wavelength" >&2; exit 1 ;;
esac

BASE_PATH="${BASE_PATH:-agent_bank/populations/Qwen3-14B_agents}"
AGENT_RANGE="${AGENT_RANGE:-0000-0199}"   # "START-END", or space-separated IDs with USE_RANGE=false
USE_RANGE="${USE_RANGE:-true}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-3}"
BATCH="${BATCH:-false}"                   # true: whole section/all headers per inference (less reliable)
INCLUDE_REASONING="${INCLUDE_REASONING:-true}"

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "${USE_RANGE}" = true ]; then
  SELECT=(--range "${AGENT_RANGE}")
else
  read -ra IDS <<< "${AGENT_RANGE}"
  SELECT=(--agents "${IDS[@]}")
fi
EXTRA=()
[ "${BATCH}" = true ] && EXTRA+=("${BATCH_FLAG}")
[ "${INCLUDE_REASONING}" = true ] || EXTRA+=(--no-reasoning)

echo "Task: ${TASK} | Base: ${BASE_PATH} | Agents: ${AGENT_RANGE} | Attempts: ${NUM_ATTEMPTS} | Model: ${GENAGENTS_MODEL:-default}"

for attempt in $(seq 1 "${NUM_ATTEMPTS}"); do
  echo ""
  echo "=== Attempt ${attempt}/${NUM_ATTEMPTS} started $(date '+%Y-%m-%d %H:%M:%S') ==="
  python3 -u "${RUNNER}" --base "${BASE_PATH}" "${SELECT[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
    || echo "WARNING: attempt ${attempt} exited with code $?; continuing"
done

echo ""
echo "=== All attempts finished $(date '+%Y-%m-%d %H:%M:%S') ==="
