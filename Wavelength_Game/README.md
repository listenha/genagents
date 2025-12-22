# Wavelength Game Module

This module provides a complete workflow for administering the Wavelength Game to generative agents, collecting their responses, and performing statistical analysis on consistency and distribution patterns.

## Overview

The Wavelength Game module enables:

1. **Game Administration**: Run Wavelength-style games on agents and store responses
2. **Consistency Analysis**: Measure agent self-consistency across multiple attempts
3. **Distribution Analysis**: Analyze how responses are distributed across different agents
4. **Response Management**: Archive and manage game response data

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
- [Data Analysis](#data-analysis)
  - [Consistency Analysis](#consistency-analysis)
  - [Distribution Analysis](#distribution-analysis)
- [File Structure](#file-structure)
- [Output Format](#output-format)
- [Design Decisions](#design-decisions)

---

## Quick Start

### 1. Run Game on Single Agent

```bash
python3 -u Wavelength_Game/run_wavelength.py --agent agent_bank/populations/gss_agents/0000
```

### 2. Run Game on Multiple Agents

```bash
# Range of agents
python3 -u Wavelength_Game/run_wavelength.py --base agent_bank/populations/gss_agents --range 0000-0049

# Specific agents
python3 -u Wavelength_Game/run_wavelength.py --base agent_bank/populations/gss_agents --agents 0000 0001 0002

# All agents
python3 -u Wavelength_Game/run_wavelength.py --base agent_bank/populations/gss_agents --all
```

### 3. Analyze Results

```bash
# Consistency analysis
python3 Wavelength_Game/analyze_consistency.py --base agent_bank/populations/gss_agents --range 0000-0049 --attempts 1-10

# Distribution analysis
python3 Wavelength_Game/analyze_distribution.py --base agent_bank/populations/gss_agents --range 0000-0049 --attempts 1-10
```

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
python3 -u Wavelength_Game/run_wavelength.py --agent agent_bank/populations/gss_agents/0000

# Range of agents
python3 -u Wavelength_Game/run_wavelength.py --base agent_bank/populations/gss_agents --range 0000-0049

# New attempt for consistency testing
python3 -u Wavelength_Game/run_wavelength.py --agent agent_bank/populations/gss_agents/0000 --new-attempt

# Faster mode (no reasoning, batch processing)
python3 -u Wavelength_Game/run_wavelength.py --base agent_bank/populations/gss_agents --range 0000-0049 --no-reasoning --batch-by-header
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

---

## Data Analysis

### Consistency Analysis

**Script**: `analyze_consistency.py`

Measures how consistent agents are across multiple game attempts using statistical reliability metrics.

#### Metrics Computed

1. **ICC(2,k)** - Intraclass Correlation Coefficient (two-way random, absolute agreement)
   - Range: 0-1
   - Interpretation: >0.75 excellent, 0.60-0.74 good, 0.40-0.59 fair, <0.40 poor

2. **ICC(3,k)** - Intraclass Correlation Coefficient (two-way mixed, consistency)
   - More lenient than ICC(2,k), allows systematic shifts

3. **Cronbach's Alpha (α)** - Internal consistency reliability
   - Range: 0-1
   - Interpretation: >0.9 excellent, 0.8-0.9 good, 0.7-0.8 acceptable

4. **Coefficient of Variation (CV)** - Relative variability
   - Lower = more consistent
   - Formula: (SD / Mean) × 100

5. **Standard Deviation (SD)** - Absolute variability
   - Lower = more consistent

6. **Range** - Max - Min across attempts
   - Lower = more consistent

#### Usage

```bash
# Basic analysis
python3 Wavelength_Game/analyze_consistency.py \
  --base agent_bank/populations/gss_agents \
  --range 0000-0049 \
  --attempts 1-10 \
  --output consistency_report.html

# Specific agents and attempts
python3 Wavelength_Game/analyze_consistency.py \
  --base agent_bank/populations/gss_agents \
  --agents 0000 0001 0002 \
  --attempts 1,2,3,5-10

# Save CSV files too
python3 Wavelength_Game/analyze_consistency.py \
  --base agent_bank/populations/gss_agents \
  --range 0000-0049 \
  --attempts 1-10 \
  --csv consistency_results/
```

#### Command-Line Options

- `--base PATH`: Base path to agent bank directory (required)
- `--agents ID1 ID2 ...`: Specific agent IDs to analyze
- `--range START-END`: Range of agent IDs (e.g., `0000-0049`)
- `--attempts RANGE`: Attempt range (e.g., `1-10`) or comma-separated (e.g., `1,2,3,5-10`)
- `--output PATH`: Output HTML file path (default: `wavelength_consistency_report.html`)
- `--csv DIR`: Also save results as CSV files in specified directory

#### Analysis Levels

Metrics are computed at multiple levels:

1. **Header-Level**: Consistency for each game header across attempts
2. **Agent-Level**: Overall consistency per agent
3. **Overall Summary**: Aggregate statistics across all headers

#### HTML Report Features

The generated HTML report includes:

- **Summary Tab**: Overall metrics dashboard, header comparison charts
- **Header-Level Tab**: Interactive heatmap (ICC by header and agent), detailed data table
- **Agent-Level Tab**: Bar chart comparing agents, agent consistency table
- **Trajectories Tab**: Line plots showing how responses change across attempts

#### Output Files

- **HTML Report**: Interactive report with Plotly visualizations (self-contained)
- **CSV Files** (if `--csv` specified):
  - `header_level.csv`: Per-header metrics
  - `agent_level.csv`: Per-agent metrics
  - `overall_summary.csv`: Overall summary statistics

### Distribution Analysis

**Script**: `analyze_distribution.py`

Analyzes how game responses are distributed across different agents, identifying patterns, diversity, and potential issues.

#### Metrics Computed

For each agent-header pair:
- **Mean Score**: Average response across attempts (central tendency)
- **Standard Deviation**: Variability across attempts (internal consistency)
- **Min/Max/Range**: Response range
- **Coefficient of Variation**: Relative variability

Then analyzes distribution of these metrics across agents:
- **Distribution Statistics**: Mean, median, SD of agent means
- **Coverage Analysis**: Which values (0-100) are used by agents
- **Diversity Metrics**: How spread out responses are
- **Outlier Detection**: Agents with unusual response patterns

#### Usage

```bash
# Basic analysis
python3 Wavelength_Game/analyze_distribution.py \
  --base agent_bank/populations/gss_agents \
  --range 0000-0049 \
  --attempts 1-10 \
  --output distribution_report.html
```

#### Command-Line Options

- `--base PATH`: Base path to agent bank directory (required)
- `--agents ID1 ID2 ...`: Specific agent IDs to analyze
- `--range START-END`: Range of agent IDs
- `--attempts RANGE`: Attempt range or comma-separated list
- `--output PATH`: Output HTML file path (default: `wavelength_distribution_report.html`)
- `--csv DIR`: Save CSV files

#### Visualizations

The HTML report includes:

1. **Box Plots**: Distribution of agent mean scores for each header
   - Shows median, IQR, outliers
   - Quick assessment of central location and spread

2. **Violin Plots**: Detailed density distribution of agent mean scores
   - Shows probability density
   - Detects multimodal distributions (polarization)

3. **Scatter Plots**: Mean vs. Standard Deviation
   - X-axis: Mean Score (central tendency)
   - Y-axis: SD (internal consistency)
   - Identifies patterns in response stability

4. **Histograms**: Distribution of agent means for each header
   - Shows frequency of different response levels

5. **Coverage Heatmap**: Which values (0-100) are used by how many agents
   - Identifies unused values or lack of diversity

6. **Edge Case Detection**: Flags headers with:
   - Lack of diversity (all agents give similar responses)
   - Unused values (no agent uses certain scale points)
   - Outliers (agents with unusual patterns)

#### Output Files

- **HTML Report**: Interactive visualizations with Plotly
- **CSV Files** (if `--csv` specified):
  - `agent_header_metrics.csv`: Per-agent, per-header metrics
  - `header_distribution.csv`: Distribution statistics per header
  - `edge_cases.csv`: Detected edge cases and issues

---

## File Structure

```
Wavelength_Game/
├── README.md                          # This file
├── game_headers.json                  # Game data (spectrum + clue pairs)
├── wavelength_response_builder.py     # Main workflow class
├── run_wavelength.py                  # CLI script to run game
├── analyze_consistency.py            # Consistency analysis script
└── analyze_distribution.py           # Distribution analysis script
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
with open('agent_bank/populations/gss_agents/0000/wavelength_responses.json') as f:
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
- Missing/null responses are excluded from analysis calculations
- Metrics are computed only when at least 2 attempts are available
- All HTML reports are self-contained (includes Plotly.js from CDN)
- Visualizations are interactive (zoom, pan, hover for details)

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

**Solution**: Ensure `game_headers.json` exists in `Wavelength_Game/` directory.

### Issue: No Responses Found

**Problem**: Analysis scripts find no data.

**Solution**: Ensure:
- Game has been run on agents (`wavelength_responses.json` exists)
- Attempt IDs match (`--attempts` flag)
- Agent folders are correctly specified

### Issue: Out of Memory (Distribution Analysis)

**Problem**: Large agent populations cause memory issues.

**Solution**: 
- Process in smaller ranges
- Use `--csv` to save intermediate results
- Process headers separately

---

## References

The Wavelength Game is based on the party game "Wavelength" by Alex Hague, Justin Vickers, and Wolfgang Warsch, published by Palm Court.

