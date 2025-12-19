#!/bin/bash
# Run interview followed by survey for a range of agents

set -e  # Exit on error

cd "$(dirname "$0")"

AGENT_RANGE="0100-0199"
BASE_PATH="agent_bank/populations/gss_agents"

# Survey settings (matching run_multiple_attempts.sh defaults)
NUM_ATTEMPTS=3
BATCH_BY_SECTION=false
INCLUDE_REASONING=true

echo "=========================================="
echo "INTERVIEW AND SURVEY PIPELINE"
echo "=========================================="
echo "Agent range: ${AGENT_RANGE}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Step 1: Run interviews
echo "STEP 1/2: Running interviews for ${AGENT_RANGE}..."
python3 -u Interview/run_interview.py --base "${BASE_PATH}" --range "${AGENT_RANGE}"

echo ""
echo "✓ Interviews completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Step 2: Run surveys
echo "STEP 2/2: Running surveys for ${AGENT_RANGE}..."
echo ""

for attempt in $(seq 1 ${NUM_ATTEMPTS}); do
    echo "--- Attempt ${attempt}/${NUM_ATTEMPTS} ---"
    
    REASONING_FLAG=""
    if [ "${INCLUDE_REASONING}" != "true" ]; then
        REASONING_FLAG="--no-reasoning"
    fi
    
    BATCH_FLAG=""
    if [ "${BATCH_BY_SECTION}" = "true" ]; then
        BATCH_FLAG="--batch-by-section"
    fi
    
    python3 -u Surveys/run_survey.py \
        --base "${BASE_PATH}" \
        --range "${AGENT_RANGE}" \
        ${BATCH_FLAG} \
        ${REASONING_FLAG}
    
    echo ""
done

echo "=========================================="
echo "✓ Pipeline completed successfully!"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="