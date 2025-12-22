#!/bin/bash
# Run Wavelength game for multiple agents with multiple attempts
# This script will run in tmux and survive SSH disconnections

# ============================================================================
# CONFIGURATION - Easy to tweak parameters
# ============================================================================

# Base path to agent bank directory
BASE_PATH="agent_bank/populations/gss_agents"

# Agent range (format: START-END, e.g., "0000-0199")
# For specific agents, use: AGENT_RANGE="0000 0001 0002" and set USE_RANGE=false
AGENT_RANGE="0100-0149"
USE_RANGE=true  # Set to false to use AGENT_RANGE as space-separated list

# Number of attempts per agent
NUM_ATTEMPTS=3

# Game options
BATCH_BY_HEADER=false   # true: faster (all headers at once), false: one header at a time (more reliable)
INCLUDE_REASONING=true  # true: store reasoning text, false: don't store (faster, smaller files)

# Optional: Different settings for different attempts
# Set to empty string to use INCLUDE_REASONING for all attempts
# Format: "attempt1:true,attempt2:true,attempt3:false"
# Example: REASONING_BY_ATTEMPT="1:true,2:true,3:false" means attempts 1-2 with reasoning, attempt 3 without
REASONING_BY_ATTEMPT=""

# Optional: Path to game headers JSON (leave empty for default)
GAME_HEADERS_PATH=""

# ============================================================================
# SCRIPT EXECUTION - Don't modify below unless you know what you're doing
# ============================================================================

# Get script directory and change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment: venv"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment: .venv"
elif [ -n "${VIRTUAL_ENV}" ]; then
    echo "Using existing virtual environment: ${VIRTUAL_ENV}"
else
    echo "WARNING: No virtual environment found. Make sure dependencies are installed."
fi
echo ""

echo "=========================================="
echo "WAVELENGTH GAME MULTIPLE ATTEMPTS RUNNER"
echo "=========================================="
echo "Base path: ${BASE_PATH}"
echo "Agent range: ${AGENT_RANGE}"
echo "Number of attempts: ${NUM_ATTEMPTS}"
echo "Batch by header: ${BATCH_BY_HEADER}"
echo "Include reasoning (default): ${INCLUDE_REASONING}"
if [ -n "${REASONING_BY_ATTEMPT}" ]; then
    echo "Reasoning by attempt: ${REASONING_BY_ATTEMPT}"
fi
if [ -n "${GAME_HEADERS_PATH}" ]; then
    echo "Game headers path: ${GAME_HEADERS_PATH}"
fi
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Build base command
BASE_CMD="python3 -u Wavelength_Game/run_wavelength.py --base ${BASE_PATH}"

if [ "${USE_RANGE}" = true ]; then
    BASE_CMD="${BASE_CMD} --range ${AGENT_RANGE}"
else
    BASE_CMD="${BASE_CMD} --agents ${AGENT_RANGE}"
fi

if [ "${BATCH_BY_HEADER}" = true ]; then
    BASE_CMD="${BASE_CMD} --batch-by-header"
fi

if [ -n "${GAME_HEADERS_PATH}" ]; then
    BASE_CMD="${BASE_CMD} --game-headers ${GAME_HEADERS_PATH}"
fi

# Always use --new-attempt to auto-increment attempt IDs
BASE_CMD="${BASE_CMD} --new-attempt"

# Function to get reasoning flag for a specific attempt
get_reasoning_flag() {
    local attempt=$1
    
    # If REASONING_BY_ATTEMPT is set, check for specific attempt
    if [ -n "${REASONING_BY_ATTEMPT}" ]; then
        # Parse REASONING_BY_ATTEMPT (format: "1:true,2:false,3:true")
        IFS=',' read -ra ATTEMPT_CONFIGS <<< "${REASONING_BY_ATTEMPT}"
        for config in "${ATTEMPT_CONFIGS[@]}"; do
            IFS=':' read -ra PARTS <<< "${config}"
            if [ "${PARTS[0]}" = "${attempt}" ]; then
                if [ "${PARTS[1]}" = "true" ]; then
                    echo ""
                else
                    echo "--no-reasoning"
                fi
                return
            fi
        done
    fi
    
    # Default: use INCLUDE_REASONING setting
    if [ "${INCLUDE_REASONING}" = true ]; then
        echo ""
    else
        echo "--no-reasoning"
    fi
}

# Run each attempt
for attempt in $(seq 1 ${NUM_ATTEMPTS}); do
    echo ""
    echo "=========================================="
    echo "ATTEMPT ${attempt}/${NUM_ATTEMPTS}"
    echo "=========================================="
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Get reasoning flag for this attempt
    REASONING_FLAG=$(get_reasoning_flag ${attempt})
    
    # Build and execute command
    CMD="${BASE_CMD} ${REASONING_FLAG}"
    
    echo "Command: ${CMD}"
    echo ""
    
    # Execute the command
    eval ${CMD}
    
    EXIT_CODE=$?
    
    echo ""
    echo "Attempt ${attempt} completed at: $(date '+%Y-%m-%d %H:%M:%S')"
    
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "WARNING: Attempt ${attempt} exited with code ${EXIT_CODE}"
        echo "Continuing with next attempt..."
    fi
    
    echo ""
done

echo "=========================================="
echo "ALL ATTEMPTS COMPLETED!"
echo "=========================================="
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="