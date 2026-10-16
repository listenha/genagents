# Surveys Module: PRE-TASK SURVEY Administration

This module administers the PRE-TASK SURVEY (BFI-10, BES-A, REI, SD3 narcissism, GDMS) to generative agents and stores their responses. Analysis of the collected responses lives in a separate repository.

## Overview

The Surveys module enables:

1. **Survey Administration**: Run standardized surveys on agents and store responses
2. **Attempt Patching**: Remove incomplete attempts and top every agent up to a target number of clean attempts
3. **Response Management**: Archive survey response data

## Table of Contents

- [Survey Instruments](#survey-instruments)
- [Quick Start](#quick-start)
- [Survey Administration](#survey-administration)
- [Response Management](#response-management)
- [File Structure](#file-structure)
- [Output Format](#output-format)
- [Design Decisions](#design-decisions)

---

## Survey Instruments

The PRE-TASK SURVEY consists of five sections (see `PRE-TASK SURVEY.md`; item counts below are those in `survey_questions.json`):

### 1. BFI-10 (Big Five Personality Inventory)
- **10 questions** measuring personality traits
- **Scale**: 1-5 Likert scale
- **Reference**: Rammstedt & John, 2007
- Measures: Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness

### 2. BES-A (Empathy Scale for Adults - Cognitive Subscale)
- **5 questions** measuring cognitive empathy
- **Scale**: 1-5 Likert scale
- **Reference**: Carré et al., 2013
- Measures: Ability to understand and recognize others' emotions

### 3. REI (Rational-Experiential Inventory - Rational Ability Subscale)
- **5 questions** measuring rational thinking ability
- **Scale**: 1-5 Likert scale
- **Reference**: Pacini & Epstein, 1999
- Measures: Logical reasoning and analytical thinking capabilities

### 4. SD3 Narcissism Subpart
- **5 questions** from the Short Dark Triad narcissism scale

### 5. GDMS (General Decision Making Style, reduced)
- **6 questions** on decision-making style

**Total**: 31 questions, all using 1-5 Likert scale where:
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
python3 Surveys/extract_survey_questions.py
```

This generates `Surveys/survey_questions.json` from `PRE-TASK SURVEY.md`.

### 2. Run Survey on Single Agent

```bash
python3 -u Surveys/run_survey.py --agent agent_bank/populations/Qwen3-14B_agents/0000
```

### 3. Run Survey on Multiple Agents

```bash
# Range of agents
python3 -u Surveys/run_survey.py --base agent_bank/populations/Qwen3-14B_agents --range 0000-0049

# Specific agents
python3 -u Surveys/run_survey.py --base agent_bank/populations/Qwen3-14B_agents --agents 0000 0001 0002

# All agents
python3 -u Surveys/run_survey.py --base agent_bank/populations/Qwen3-14B_agents --all
```

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
python3 -u Surveys/run_survey.py --agent agent_bank/populations/Qwen3-14B_agents/0000

# Range of agents, specific sections
python3 -u Surveys/run_survey.py --base agent_bank/populations/Qwen3-14B_agents --range 0000-0049 --sections BFI-10 REI

# New attempt for consistency testing
python3 -u Surveys/run_survey.py --agent agent_bank/populations/Qwen3-14B_agents/0000 --new-attempt

# Faster mode (no reasoning, batch processing)
python3 -u Surveys/run_survey.py --base agent_bank/populations/Qwen3-14B_agents --range 0000-0049 --no-reasoning --batch-by-section

# Multiple attempts (repo-root script; settings via environment)
NUM_ATTEMPTS=3 BASE_PATH=agent_bank/populations/Qwen3-14B_agents ./run_multiple_attempts.sh survey
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

### Patching to a Target Number of Clean Attempts

`run_survey_patch_attempts.py` is how the TUX populations were brought to exactly 3 attempts per agent. For every agent it removes "dirty" attempts (any missing or `null` response), renumbers the rest, then runs new attempts (one question per inference) until the target is reached.

```bash
# Model chosen via environment (see .env.example) ...
GENAGENTS_MODEL=14b python3 -u Surveys/run_survey_patch_attempts.py \
  --base agent_bank/populations/Qwen3-14B_agents --target-attempts 3

# ... or via flag; pin to one GPU to run several populations in parallel
CUDA_VISIBLE_DEVICES=1 python3 -u Surveys/run_survey_patch_attempts.py \
  --base agent_bank/populations/Llama-3.1-8B_agents --target-attempts 3 --model-choice llama3.1-8b

# Report only
python3 Surveys/run_survey_patch_attempts.py --base agent_bank/populations/Qwen3-14B_agents --dry-run
```

---

## Response Management

### Archiving Responses

**Script**: `archive_survey_responses.py`

Move each agent's current `survey_responses.json` (and `survey_distilled_responses.json`, if present) into `<agent>/archive/<timestamp>_*.json`, so a fresh round of attempts can start from an empty file.

```bash
python3 Surveys/archive_survey_responses.py \
  --base agent_bank/populations/my_agents \
  --range 0000-0049
```

## File Structure

```
Surveys/
├── README.md                          # This file
├── PRE-TASK SURVEY.md                 # Original survey document
├── survey_questions.json               # Structured survey data (generated)
├── extract_survey_questions.py        # Extract questions from markdown
├── survey_response_builder.py          # Main workflow class
├── run_survey.py                      # CLI script to run surveys
├── run_survey_patch_attempts.py       # Drop dirty attempts, top up to N clean attempts
├── archive_survey_responses.py        # Archive responses into <agent>/archive/
└── PROMPT_EXAMPLE.md                  # Worked example of an assembled survey prompt
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
with open('agent_bank/populations/Qwen3-14B_agents/0000/survey_responses.json') as f:
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

## References

- **BFI-10**: Rammstedt, B., & John, O. P. (2007). Measuring personality in one minute or less: A 10-item short version of the Big Five Inventory in English and German. *Journal of Research in Personality*, 41(1), 203-212.

- **BES-A**: Carré, A., Stefaniak, N., D'Ambrosio, F., Bensalah, L., & Besche-Richard, C. (2013). The Basic Empathy Scale in Adults (BES-A): Factor structure of a revised form. *Psychological Assessment*, 25(3), 679-691.

- **REI**: Pacini, R., & Epstein, S. (1999). The relation of rational and experiential information processing styles to personality, basic beliefs, and the ratio-bias phenomenon. *Journal of Personality and Social Psychology*, 76(6), 972-987.
