# Surveys Module: PRE-TASK SURVEY Administration and Analysis

This module provides a complete workflow for administering the PRE-TASK SURVEY to generative agents, collecting responses, and performing statistical analysis on consistency and distribution patterns.

## Overview

The Surveys module enables:

1. **Survey Administration**: Run standardized surveys on agents and store responses
2. **Consistency Analysis**: Measure agent self-consistency across multiple attempts
3. **Distribution Analysis**: Analyze how responses are distributed across different agents
4. **Response Management**: Archive, translate, and manage survey response data

## Table of Contents

- [Survey Instruments](#survey-instruments)
- [Quick Start](#quick-start)
- [Survey Administration](#survey-administration)
- [Data Analysis](#data-analysis)
  - [Consistency Analysis](#consistency-analysis)
  - [Distribution Analysis](#distribution-analysis)
- [Response Management](#response-management)
- [File Structure](#file-structure)
- [Output Format](#output-format)
- [Design Decisions](#design-decisions)

---

## Survey Instruments

The PRE-TASK SURVEY consists of three standardized psychological instruments:

### 1. BFI-10 (Big Five Personality Inventory)
- **10 questions** measuring personality traits
- **Scale**: 1-5 Likert scale
- **Reference**: Rammstedt & John, 2007
- Measures: Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness

### 2. BES-A (Empathy Scale for Adults - Cognitive Subscale)
- **9 questions** measuring cognitive empathy
- **Scale**: 1-5 Likert scale
- **Reference**: Carré et al., 2013
- Measures: Ability to understand and recognize others' emotions

### 3. REI (Rational-Experiential Inventory - Rational Ability Subscale)
- **10 questions** measuring rational thinking ability
- **Scale**: 1-5 Likert scale
- **Reference**: Pacini & Epstein, 1999
- Measures: Logical reasoning and analytical thinking capabilities

**Total**: 29 questions, all using 1-5 Likert scale where:
- 1 = Disagree Strongly
- 2 = Disagree a Little
- 3 = Neither Agree nor Disagree
- 4 = Agree a Little
- 5 = Agree Strongly

See `PRE-TASK SURVEY.md` for the complete survey text.

---

## Quick Start

### 1. Extract Survey Questions

First, extract survey questions from the markdown file:

```bash
cd /taiga/common_resources/yueshen7/genagents
python3 Surveys/extract_survey_questions.py
```

This generates `Surveys/survey_questions.json` from `PRE-TASK SURVEY.md`.

### 2. Run Survey on Single Agent

```bash
python3 -u Surveys/run_survey.py --agent agent_bank/populations/gss_agents/0000
```

### 3. Run Survey on Multiple Agents

```bash
# Range of agents
python3 -u Surveys/run_survey.py --base agent_bank/populations/gss_agents --range 0000-0049

# Specific agents
python3 -u Surveys/run_survey.py --base agent_bank/populations/gss_agents --agents 0000 0001 0002

# All agents
python3 -u Surveys/run_survey.py --base agent_bank/populations/gss_agents --all
```

### 4. Analyze Results

```bash
# Consistency analysis
python3 Surveys/analyze_consistency.py --base agent_bank/populations/gss_agents --range 0000-0049 --attempts 1-10

# Distribution analysis
python3 Surveys/analyze_distribution.py --base agent_bank/populations/gss_agents --range 0000-0049 --attempts 1-10
```

---

## Survey Administration

### Running Surveys

The main entry point is `run_survey.py`, which administers surveys to agents using the `SurveyResponseBuilder` class.

#### Command-Line Options

**Input Options:**
- `--agent PATH`: Single agent folder path
- `--base PATH`: Base path to agent bank (requires `--agents`, `--range`, or `--all`)
- `--agents ID1 ID2 ...`: Specific agent IDs to process
- `--range START-END`: Range of agent IDs (e.g., `0000-0049`)
- `--all`: Process all agents in base directory

**Survey Options:**
- `--survey PATH`: Custom survey questions JSON (default: `Surveys/survey_questions.json`)
- `--sections SECTION1 SECTION2 ...`: Specific sections only (default: all sections)
- `--new-attempt`: Create new attempt (auto-increments `attempt_id`)
- `--attempt-id N`: Specific attempt ID to use
- `--no-reasoning`: Don't store reasoning text (faster, smaller files)
- `--batch-by-section`: Process all questions in a section at once (faster)

**Other Options:**
- `--dry-run`: Preview what would be processed without running

#### Examples

```bash
# Single agent, all sections
python3 -u Surveys/run_survey.py --agent agent_bank/populations/gss_agents/0000

# Range of agents, specific sections
python3 -u Surveys/run_survey.py --base agent_bank/populations/gss_agents --range 0000-0049 --sections BFI-10 REI

# New attempt for consistency testing
python3 -u Surveys/run_survey.py --agent agent_bank/populations/gss_agents/0000 --new-attempt

# Faster mode (no reasoning, batch processing)
python3 -u Surveys/run_survey.py --base agent_bank/populations/gss_agents --range 0000-0049 --no-reasoning --batch-by-section

# Multiple attempts script
bash Surveys/run_multiple_attempts.sh
```

### How It Works

1. **Load Agent**: Reads `scratch.json` and `memory_stream/nodes.json` from agent folder
2. **Retrieve Context**: Uses agent's memory stream to retrieve relevant memories for each question
3. **Generate Responses**: Calls `agent.numerical_resp()` with survey questions (1-5 scale)
4. **Store Responses**: Saves responses to `{agent_folder}/survey_responses.json` (NOT in memory stream)

### Response Generation Process

The survey uses the existing `numerical_resp` prompt template, which:
- Takes agent's scratchpad (demographics) + retrieved memories (interview Q&A)
- Formats questions as "Q: [question]\nRange: [1, 5]"
- Requests LLM to predict responses based on the interview transcript
- Returns numerical responses (1-5) with optional reasoning

**Note**: The prompt template references "interview transcript" but receives scratch.json + memory stream, which works perfectly for agents with interview memories.

### Multiple Attempts

Each agent can have multiple survey attempts for consistency testing:

- Each attempt gets a unique `attempt_id` (auto-incremented)
- All attempts stored in the same `survey_responses.json` file
- Easy to track consistency across attempts
- Use `--new-attempt` to create a new attempt

---

## Data Analysis

### Consistency Analysis

**Script**: `analyze_consistency.py`

Measures how consistent agents are across multiple survey attempts using statistical reliability metrics.

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
python3 Surveys/analyze_consistency.py \
  --base agent_bank/populations/gss_agents \
  --range 0000-0049 \
  --attempts 1-10 \
  --output consistency_report.html

# Specific agents and attempts
python3 Surveys/analyze_consistency.py \
  --base agent_bank/populations/gss_agents \
  --agents 0000 0001 0002 \
  --attempts 1,2,3,5-10

# Save CSV files too
python3 Surveys/analyze_consistency.py \
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
- `--output PATH`: Output HTML file path (default: `consistency_report.html`)
- `--csv DIR`: Also save results as CSV files in specified directory

#### Analysis Levels

Metrics are computed at multiple levels:

1. **Question-Level**: Consistency for each question across attempts
2. **Section-Level**: Consistency for each section (BFI-10, BES-A, REI)
3. **Agent-Level**: Overall consistency per agent
4. **Overall Summary**: Aggregate statistics by section

#### HTML Report Features

The generated HTML report includes:

- **Summary Tab**: Overall metrics dashboard, section comparison charts
- **Question-Level Tab**: Interactive heatmap (ICC by question and agent), detailed data table
- **Agent-Level Tab**: Bar chart comparing agents, agent consistency table
- **Section-Level Tab**: Section comparison charts, section summary table
- **Trajectories Tab**: Line plots showing how responses change across attempts

#### Output Files

- **HTML Report**: Interactive report with Plotly visualizations (self-contained)
- **CSV Files** (if `--csv` specified):
  - `question_level.csv`: Per-question metrics
  - `section_level.csv`: Per-section metrics
  - `agent_level.csv`: Per-agent metrics
  - `overall_section.csv`: Overall section summary

### Distribution Analysis

**Script**: `analyze_distribution.py`

Analyzes how survey responses are distributed across different agents, identifying patterns, diversity, and potential issues.

#### Metrics Computed

For each agent-question pair:
- **Mean Score**: Average response across attempts (central tendency)
- **Standard Deviation**: Variability across attempts (internal consistency)
- **Min/Max/Range**: Response range
- **Coefficient of Variation**: Relative variability

Then analyzes distribution of these metrics across agents:
- **Distribution Statistics**: Mean, median, SD of agent means
- **Coverage Analysis**: Which values (1-5) are used by agents
- **Diversity Metrics**: How spread out responses are
- **Outlier Detection**: Agents with unusual response patterns

#### Usage

```bash
# Basic analysis
python3 Surveys/analyze_distribution.py \
  --base agent_bank/populations/gss_agents \
  --range 0000-0049 \
  --attempts 1-10 \
  --output distribution_report.html

# Include distilled scores
python3 Surveys/analyze_distribution.py \
  --base agent_bank/populations/gss_agents \
  --range 0000-0049 \
  --attempts 1-10 \
  --include-distilled
```

#### Command-Line Options

- `--base PATH`: Base path to agent bank directory (required)
- `--agents ID1 ID2 ...`: Specific agent IDs to analyze
- `--range START-END`: Range of agent IDs
- `--attempts RANGE`: Attempt range or comma-separated list
- `--output PATH`: Output HTML file path (default: `distribution_report.html`)
- `--include-distilled`: Also analyze distilled scores if available
- `--csv DIR`: Save CSV files

#### Visualizations

The HTML report includes:

1. **Box Plots**: Distribution of agent mean scores for each question
   - Shows median, IQR, outliers
   - Quick assessment of central location and spread

2. **Violin Plots**: Detailed density distribution of agent mean scores
   - Shows probability density
   - Detects multimodal distributions (polarization)

3. **Scatter Plots**: Mean vs. Standard Deviation
   - X-axis: Mean Score (central tendency)
   - Y-axis: SD (internal consistency)
   - Identifies patterns in response stability

4. **Histograms**: Distribution of agent means for each question
   - Shows frequency of different response levels

5. **Coverage Heatmap**: Which values (1-5) are used by how many agents
   - Identifies unused values or lack of diversity

6. **Edge Case Detection**: Flags questions with:
   - Lack of diversity (all agents give similar responses)
   - Unused values (no agent uses certain scale points)
   - Outliers (agents with unusual patterns)

#### Output Files

- **HTML Report**: Interactive visualizations with Plotly
- **CSV Files** (if `--csv` specified):
  - `agent_question_metrics.csv`: Per-agent, per-question metrics
  - `question_distribution.csv`: Distribution statistics per question
  - `edge_cases.csv`: Detected edge cases and issues

---

## Response Management

### Archiving Responses

**Script**: `archive_survey_responses.py`

Archive survey responses to a separate location for backup or analysis.

```bash
python3 Surveys/archive_survey_responses.py \
  --base agent_bank/populations/gss_agents \
  --range 0000-0049 \
  --archive-dir archived_responses/
```

### Translating Responses

**Script**: `translate_responses.py`

Translate survey responses to different languages (useful for international research).

```bash
python3 Surveys/translate_responses.py \
  --base agent_bank/populations/gss_agents \
  --range 0000-0049 \
  --target-language es  # Spanish
```

---

## File Structure

```
Surveys/
├── README.md                          # This file
├── PRE-TASK SURVEY.md                 # Original survey document
├── survey_questions.json               # Structured survey data (generated)
├── extract_survey_questions.py        # Extract questions from markdown
├── survey_response_builder.py          # Main workflow class
├── run_survey.py                      # CLI script to run surveys
├── analyze_consistency.py             # Consistency analysis script
├── analyze_distribution.py            # Distribution analysis script
├── archive_survey_responses.py        # Archive responses
├── translate_responses.py             # Translate responses
├── run_multiple_attempts.sh           # Batch script for multiple attempts
└── Survey Translation/                # Translation files
    ├── BFI-10.md
    ├── BES-A.md
    └── REI.md
```

---

## Output Format

Each agent's responses are stored in `{agent_folder}/survey_responses.json`:

```json
{
  "survey_metadata": {
    "survey_name": "PRE-TASK SURVEY",
    "survey_version": "1.0",
    "extraction_date": "2025-12-08T02:34:26.308359"
  },
  "agent_metadata": {
    "agent_id": "0000",
    "agent_name": "Gabriela Johnson"
  },
  "attempts": [
    {
      "attempt_id": 1,
      "timestamp": "2025-12-09T02:05:45.035075",
      "sections": {
        "BFI-10": {
          "responses": {
            "BFI-10_1": 3,
            "BFI-10_2": 4,
            ...
          },
          "raw_responses": {
            "BFI-10_1": "Response: 3. Reasoning: ...",
            ...
          }
        },
        "BES-A": { ... },
        "REI": { ... }
      },
      "completion_status": "completed"
    }
  ]
}
```

### Data Structure

- **`responses`**: Clean numerical responses (1-5) for easy analysis
- **`raw_responses`**: Full LLM reasoning text (optional, can be disabled with `--no-reasoning`)
- **`attempts`**: Array of all attempts, each with unique `attempt_id`

### Converting to DataFrame

```python
import json
import pandas as pd

# Load responses
with open('agent_bank/populations/gss_agents/0000/survey_responses.json') as f:
    data = json.load(f)

# Extract first attempt
attempt = data['attempts'][0]

# Convert to DataFrame
rows = []
for section_id, section_data in attempt['sections'].items():
    for question_id, response in section_data['responses'].items():
        rows.append({
            'agent_id': data['agent_metadata']['agent_id'],
            'attempt_id': attempt['attempt_id'],
            'section': section_id,
            'question_id': question_id,
            'response': response
        })

df = pd.DataFrame(rows)
```

---

## Design Decisions

### Memory Stream

- Survey interactions are **NOT** saved to memory stream
- Only final responses are stored in `survey_responses.json`
- Agent answers based on existing memory (scratch.json + memory stream), but survey itself isn't remembered
- This prevents survey responses from influencing future survey attempts

### Using Existing Template

- Uses `agent.numerical_resp()` which calls the existing `numerical_resp` prompt template
- Template says "interview transcript" but receives: scratch.json + retrieved memories (including interview Q&A)
- Works perfectly for agents with interview memories
- No need for survey-specific prompt templates

### Multiple Attempts

- Each attempt gets unique `attempt_id` (auto-incremented)
- All attempts stored in same JSON file
- Easy to track consistency across attempts
- Use `--new-attempt` flag to create new attempt

### Section Prefixes

- Section prefixes (e.g., "I see myself as someone who…") are included with every question
- Questions are stored with full text including prefix
- Ensures context is preserved

### Reasoning Storage

- By default, reasoning text is stored in `raw_responses`
- Use `--no-reasoning` flag to skip (faster, smaller files)
- Reasoning includes LLM's interpretation and reasoning for each response

### Batch Processing

- `--batch-by-section` processes all questions in a section at once
- More efficient (fewer API calls)
- Faster execution for large agent populations

### Token Management

- Automatically calculates `max_tokens` based on number of questions
- Formula: `max(2000, num_questions * 300)`
- Prevents truncation of responses
- See `PROMPT_EXAMPLE.md` for details on prompt structure

---

## Notes

- Use `python3 -u` flag for unbuffered output (real-time progress)
- Survey requires all sections to be completed (configurable in code)
- Each section is administered as a batch to the LLM (efficient)
- Responses are automatically validated (1-5 scale)
- Missing/null responses are excluded from analysis calculations
- Metrics are computed only when at least 2 attempts are available
- All HTML reports are self-contained (includes Plotly.js from CDN)
- Visualizations are interactive (zoom, pan, hover for details)

---

## Troubleshooting

### Issue: Truncated Responses

**Problem**: Last questions have `null` responses.

**Solution**: The system automatically calculates `max_tokens` based on number of questions. If issues persist, check:
- LLM model capacity
- Network connectivity
- API rate limits

### Issue: Missing Survey Questions

**Problem**: `survey_questions.json` not found.

**Solution**: Run `extract_survey_questions.py` first:
```bash
python3 Surveys/extract_survey_questions.py
```

### Issue: No Responses Found

**Problem**: Analysis scripts find no data.

**Solution**: Ensure:
- Survey has been run on agents (`survey_responses.json` exists)
- Attempt IDs match (`--attempts` flag)
- Agent folders are correctly specified

### Issue: Out of Memory (Distribution Analysis)

**Problem**: Large agent populations cause memory issues.

**Solution**: 
- Process in smaller ranges
- Use `--csv` to save intermediate results
- Process sections separately

---

## References

- **BFI-10**: Rammstedt, B., & John, O. P. (2007). Measuring personality in one minute or less: A 10-item short version of the Big Five Inventory in English and German. *Journal of Research in Personality*, 41(1), 203-212.

- **BES-A**: Carré, A., Stefaniak, N., D'Ambrosio, F., Bensalah, L., & Besche-Richard, C. (2013). The Basic Empathy Scale in Adults (BES-A): Factor structure of a revised form. *Psychological Assessment*, 25(3), 679-691.

- **REI**: Pacini, R., & Epstein, S. (1999). The relation of rational and experiential information processing styles to personality, basic beliefs, and the ratio-bias phenomenon. *Journal of Personality and Social Psychology*, 76(6), 972-987.
