# Wavelength Game Module

This module administers the Wavelength spectrum-placement game used in the TUX study to generative agents and stores their responses. Analysis of the collected responses lives in a separate repository.

## Overview

The Wavelength Game module enables:

1. **Game Administration**: Run Wavelength-style games on agents and store responses
2. **Attempt Patching**: Remove incomplete attempts and top every agent up to a target number of clean attempts

## Wavelength Game Rules

The Wavelength Game is a spectrum-based game where:

- A **spectrum** is defined between two opposites (e.g., "Hot — Cold")
- A **clue** is given (e.g., "Coffee")
- Each player places a marker on the spectrum (0-100) where they think the clue belongs
  - 0 means completely on the left side of the spectrum
  - 100 means completely on the right side of the spectrum
- Players provide reasoning for their selection to ensure choice-accuracy and internal consistency

**Example**:
- Spectrum: "Hot — Cold"
- Clue: "Coffee"
- A player who thinks coffee is generally hot might choose a value closer to 0 (Hot)
- A player who favors cold coffee might choose a value closer to 100 (Cold)

## Table of Contents

- [Quick Start](#quick-start)
- [Game Administration](#game-administration)
- [Game Headers (Source of Truth)](#game-headers-source-of-truth)
- [File Structure](#file-structure)
- [Output Format](#output-format)
- [Design Decisions](#design-decisions)

---

## Quick Start

### 1. Extract Game Headers (if needed)

If you have edited `Game Headers.csv` or do not yet have `game_headers.json`, run:

```bash
python3 Wavelength_Game/extract_game_headers.py
```

See [Game Headers (Source of Truth)](#game-headers-source-of-truth) for details.

### 2. Run Game on Single Agent

```bash
python3 -u Wavelength_Game/run_wavelength.py --agent agent_bank/populations/Qwen3-14B_agents/0000
```

### 3. Run Game on Multiple Agents

```bash
# Range of agents
python3 -u Wavelength_Game/run_wavelength.py --base agent_bank/populations/Qwen3-14B_agents --range 0000-0049

# Specific agents
python3 -u Wavelength_Game/run_wavelength.py --base agent_bank/populations/Qwen3-14B_agents --agents 0000 0001 0002

# All agents
python3 -u Wavelength_Game/run_wavelength.py --base agent_bank/populations/Qwen3-14B_agents --all
```

## Game Headers (Source of Truth)

The set of game headers (spectrum + clue pairs) that agents answer is defined in **`Wavelength_Game/Game Headers.csv`**. That CSV is the **source of truth** for the game.

- **CSV columns**: `Cue`, `Spectrum Left`, `Spectrum Right`
- **Derived file**: `game_headers.json` is generated from the CSV and is what `run_wavelength.py` and `WavelengthResponseBuilder` load at runtime.

### Extracting game_headers.json

After editing `Game Headers.csv`, regenerate the JSON:

```bash
python3 Wavelength_Game/extract_game_headers.py
```

This reads `Wavelength_Game/Game Headers.csv` and writes `Wavelength_Game/game_headers.json`. You must run this step before running the game if you changed the CSV.

---

## Game Administration

### Running Games

The main entry point is `run_wavelength.py`, which administers the Wavelength Game to agents using the `WavelengthResponseBuilder` class.

#### Command-Line Options

**Input Options:**
- `--agent PATH`: Single agent folder path
- `--base PATH`: Base path to agent bank (requires `--agents`, `--range`, or `--all`)
- `--agents ID1 ID2 ...`: Specific agent IDs to process
- `--range START-END`: Range of agent IDs (e.g., `0000-0049`)
- `--all`: Process all agents in base directory

**Game Options:**
- `--game-headers PATH`: Custom game headers JSON (default: `Wavelength_Game/game_headers.json`)
- `--new-attempt`: Create new attempt (auto-increments `attempt_id`)
- `--attempt-id N`: Specific attempt ID to use
- `--no-reasoning`: Don't store reasoning text (faster, smaller files)
- `--batch-by-header`: Process all headers at once (faster)

**Other Options:**
- `--dry-run`: Preview what would be processed without running

#### Examples

```bash
# Single agent, all headers
python3 -u Wavelength_Game/run_wavelength.py --agent agent_bank/populations/Qwen3-14B_agents/0000

# Range of agents
python3 -u Wavelength_Game/run_wavelength.py --base agent_bank/populations/Qwen3-14B_agents --range 0000-0049

# New attempt for consistency testing
python3 -u Wavelength_Game/run_wavelength.py --agent agent_bank/populations/Qwen3-14B_agents/0000 --new-attempt

# Faster mode (no reasoning, batch processing)
python3 -u Wavelength_Game/run_wavelength.py --base agent_bank/populations/Qwen3-14B_agents --range 0000-0049 --no-reasoning --batch-by-header
```

### How It Works

1. **Load Agent**: Reads `scratch.json` and `memory_stream/nodes.json` from agent folder
2. **Retrieve Context**: Uses agent's memory stream to retrieve relevant memories for each game header
3. **Generate Responses**: Calls `agent.numerical_resp()` with Wavelength-specific prompt template (0-100 scale)
4. **Store Responses**: Saves responses to `{agent_folder}/wavelength_responses.json` (NOT in memory stream)

### Response Generation Process

The game uses a Wavelength-specific prompt template that:
- Frames the task as a Wavelength-style game
- Explains the spectrum concept (left vs right poles)
- Asks agent to place marker on 0-100 scale based on their perspective
- Requests reasoning for their choice
- Returns numerical responses (0-100) with reasoning

**Note**: The prompt template is specifically designed for Wavelength game context, not "interview transcript" framing.

### Multiple Attempts

Each agent can have multiple game attempts for consistency testing:

- Each attempt gets a unique `attempt_id` (auto-incremented)
- All attempts stored in the same `wavelength_responses.json` file
- Easy to track consistency across attempts
- Use `--new-attempt` to create a new attempt

### Patching to a Target Number of Clean Attempts

`run_wavelength_patch_attempts.py` is how the TUX populations were brought to exactly 3 attempts per agent. For every agent it removes "dirty" attempts (any header missing or `null`), renumbers the rest, then runs new attempts (one header per inference) until the target is reached.

```bash
GENAGENTS_MODEL=14b python3 -u Wavelength_Game/run_wavelength_patch_attempts.py \
  --base agent_bank/populations/Qwen3-14B_agents --target-attempts 3

# Equivalent, selecting the model by flag and pinning one GPU
CUDA_VISIBLE_DEVICES=0 python3 -u Wavelength_Game/run_wavelength_patch_attempts.py \
  --base agent_bank/populations/Mistral-Nemo_agents --target-attempts 3 --model-choice mistral-nemo-2407
```

---

## File Structure

```
Wavelength_Game/
├── README.md                          # This file
├── Game Headers.csv                   # Source of truth for game headers (Cue, Spectrum Left, Spectrum Right)
├── extract_game_headers.py            # Converts Game Headers.csv → game_headers.json
├── game_headers.json                  # Game data used at runtime (generated from CSV)
├── wavelength_response_builder.py     # Main workflow class
├── run_wavelength.py                  # CLI script to run game
└── run_wavelength_patch_attempts.py   # Drop dirty attempts, top up to N clean attempts
```

---

## Output Format

Each agent's responses are stored in `{agent_folder}/wavelength_responses.json`:

```json
{
  "game_metadata": {
    "game_name": "Wavelength Game",
    "game_version": "1.0",
    "extraction_date": "2025-01-15T12:00:00.000000"
  },
  "agent_metadata": {
    "agent_id": "0000",
    "agent_name": "Gabriela Johnson"
  },
  "attempts": [
    {
      "attempt_id": 1,
      "timestamp": "2025-01-15T12:05:45.035075",
      "headers": {
        "header_0": {
          "spectrum": {
            "left": "Scared to death",
            "right": "bored to death"
          },
          "cue": "Horror Movie",
          "response": 11,
          "raw_response": {
            "Range Interpretation": {
              "0 (Scared to death)": "...",
              "100 (bored to death)": "..."
            },
            "Reasoning": "...",
            "Response": 11
          }
        },
        "header_1": {
          ...
        }
      },
      "completion_status": "completed"
    }
  ]
}
```

### Data Structure

- **`response`**: Clean numerical response (0-100) for easy analysis
- **`raw_response`**: Full LLM reasoning text (optional, can be disabled with `--no-reasoning`)
- **`attempts`**: Array of all attempts, each with unique `attempt_id`
- **`headers`**: Dictionary keyed by `header_N` where N is the index from `game_headers.json`

### Converting to DataFrame

```python
import json
import pandas as pd

# Load responses
with open('agent_bank/populations/Qwen3-14B_agents/0000/wavelength_responses.json') as f:
    data = json.load(f)

# Extract first attempt
attempt = data['attempts'][0]

# Convert to DataFrame
rows = []
for header_id, header_data in attempt['headers'].items():
    rows.append({
        'agent_id': data['agent_metadata']['agent_id'],
        'attempt_id': attempt['attempt_id'],
        'header_id': header_id,
        'spectrum_left': header_data['spectrum']['left'],
        'spectrum_right': header_data['spectrum']['right'],
        'cue': header_data['cue'],
        'response': header_data['response']
    })

df = pd.DataFrame(rows)
```

---

## Design Decisions

### Memory Stream

- Game interactions are **NOT** saved to memory stream
- Only final responses are stored in `wavelength_responses.json`
- Agent answers based on existing memory (scratch.json + memory stream), but game itself isn't remembered
- This prevents game responses from influencing future game attempts

### Wavelength-Specific Prompt Template

- Uses custom prompt template designed for Wavelength game context
- Template frames task as a Wavelength-style game (not "interview transcript")
- Explains spectrum concept clearly
- Requests reasoning to ensure choice-accuracy and internal consistency

### Multiple Attempts

- Each attempt gets unique `attempt_id` (auto-incremented)
- All attempts stored in same JSON file
- Easy to track consistency across attempts
- Use `--new-attempt` flag to create new attempt

### Reasoning Storage

- By default, reasoning text is stored in `raw_response`
- Use `--no-reasoning` flag to skip (faster, smaller files)
- Reasoning includes LLM's interpretation and reasoning for each response

### Batch Processing

- `--batch-by-header` processes all headers at once
- More efficient (fewer API calls)
- Faster execution for large agent populations
- Default: one header at a time (more reliable)

### Response Scale

- Uses 0-100 integer scale (not float)
- 0 = completely on left side of spectrum
- 100 = completely on right side of spectrum
- Matches Wavelength game mechanics

---

## Notes

- Use `python3 -u` flag for unbuffered output (real-time progress)
- Game requires all headers to be completed (configurable in code)
- Each header is administered individually by default (can batch with `--batch-by-header`)
- Responses are automatically validated (0-100 scale)

---

## Troubleshooting

### Issue: Truncated Responses

**Problem**: Last headers have `null` responses.

**Solution**: The system automatically calculates `max_tokens` based on number of headers. If issues persist, check:
- LLM model capacity
- Network connectivity
- API rate limits
- Use `--batch-by-header` with caution (may cause truncation)

### Issue: Missing Game Headers

**Problem**: `game_headers.json` not found.

**Solution**: Run `python3 Wavelength_Game/extract_game_headers.py` to generate it from `Game Headers.csv`. Ensure `Game Headers.csv` exists in `Wavelength_Game/`.

## References

The Wavelength Game is based on the party game "Wavelength" by Alex Hague, Justin Vickers, and Wolfgang Warsch, published by Palm Court.

