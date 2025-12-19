# Interview Memory Builder

A modular system to populate agent memory streams through semi-structured interviews, starting from demographic profiles in `scratch.json`, following the American Voices Project interview protocol.

## Overview

The Interview Memory Builder conducts interviews with agents to build rich, detailed personal profiles. Each question-answer pair is saved as an observation memory, and reflections are triggered based on configurable strategies.

## Components

### 1. Interview Script Extraction
**File:** `extract_interview_script.py`

Extracts interview questions from the example agent's memory stream nodes and saves them as a JSON file.

**Usage:**
```bash
python3 agent_bank/scripts/extract_interview_script.py
```

This creates `agent_bank/scripts/interview_script.json` with all interview questions.

### 2. Interview Memory Builder
**File:** `environment/interview/interview_memory_builder.py`

Main class for conducting interviews and populating memory streams.

**Basic Usage:**
```python
from environment.interview.interview_memory_builder import InterviewMemoryBuilder

# Initialize builder
builder = InterviewMemoryBuilder(
    interview_script_path="agent_bank/scripts/interview_script.json",
    config={
        "reflection": {
            "periodic": {"enabled": True, "interval": 5},
            "high_importance": {"enabled": True, "threshold": 80},
            "manual": {"enabled": True, "question_ids": [1, 10, 20]}
        },
        "save_interval": 10
    }
)

# Build memory for a single agent
result = builder.build_memory("agent_bank/populations/gss_agents/0000")
print(f"Questions answered: {result['questions_answered']}")
print(f"Memories created: {result['memories_created']}")
print(f"Reflections triggered: {result['reflections_triggered']}")
```

**Batch Processing:**
```python
# Process multiple agents
agent_folders = [
    "agent_bank/populations/gss_agents/0000",
    "agent_bank/populations/gss_agents/0001",
    # ...
]

results = builder.build_memory_batch(agent_folders)
for result in results:
    if result["status"] == "success":
        print(f"✓ {result['agent_folder']}: {result['questions_answered']} questions")
    else:
        print(f"✗ {result['agent_folder']}: {result['error']}")
```

## Configuration

### Reflection Strategies

1. **Periodic Reflection**
   - Triggers every N questions
   - Default: every 5 questions
   - Anchor: "recent interview topics from questions X to Y"

2. **High-Importance Reflection**
   - Triggers when a memory's importance score exceeds threshold
   - Default threshold: 80
   - Anchor: "important topic: [question text]"

3. **Manual Reflection**
   - Triggers at specific question IDs
   - Useful for major topic transitions
   - Anchor: "key topic: [question text]"

### Default Configuration

```python
{
    "reflection": {
        "periodic": {"enabled": True, "interval": 5},
        "high_importance": {"enabled": True, "threshold": 80},
        "manual": {"enabled": True, "question_ids": []}
    },
    "context": "You are developing a detailed personal profile...",
    "interviewer_name": "Interviewer",
    "save_interval": 10,
    "prompt_template": "simulation_engine/prompt_template/generative_agent/interaction/utternace/interview_v1.txt"
}
```

## Memory Format

Each question-answer pair is saved as an observation memory with the format:

```
Interviewer: [question]

[Agent Name]: [response]

```

The memory is assigned:
- `node_type`: "observation"
- `time_step`: Sequential counter (starting from 1, node 0 is empty)
- `importance`: LLM-generated score (0-100)
- `created`: Same as time_step
- `last_retrieved`: Same as time_step initially

## Progress Tracking

The builder automatically saves progress every N questions (configurable via `save_interval`). If interrupted, the interview can be resumed from the last saved checkpoint.

Progress is stored in `{agent_folder}/interview_progress.json`:
```json
{
  "last_question_idx": 10,
  "last_time_step": 11
}
```

## Running Interviews

### Quick Start

Run a complete interview on a single agent:

```bash
python3 agent_bank/scripts/run_interview.py --agent agent_bank/populations/gss_agents/0000
```

### Common Usage Examples

**Test with limited questions:**
```bash
python3 agent_bank/scripts/run_interview.py --agent agent_bank/populations/gss_agents/0000 --limit 10
```

**Use custom configuration:**
```bash
python3 agent_bank/scripts/run_interview.py --agent agent_bank/populations/gss_agents/0000 --config interview_config_example.json
```

**Process multiple specific agents:**
```bash
python3 agent_bank/scripts/run_interview.py --base agent_bank/populations/gss_agents --agents 0000 0001 0002
```

**Process all agents in directory:**
```bash
python3 agent_bank/scripts/run_interview.py --base agent_bank/populations/gss_agents --all
```

**Dry run (preview without executing):**
```bash
python3 agent_bank/scripts/run_interview.py --agent agent_bank/populations/gss_agents/0000 --dry-run
```

### Command-Line Options

- `--agent PATH`: Process a single agent folder
- `--base PATH`: Base path to agent bank directory
- `--agents ID1 ID2 ...`: Process specific agent IDs (requires --base)
- `--all`: Process all agents in base directory (requires --base)
- `--config PATH`: Path to JSON configuration file
- `--script PATH`: Path to interview script (default: interview_script.json)
- `--limit N`: Limit number of questions (for testing)
- `--output PATH`: Output directory (default: saves to agent folder)
- `--dry-run`: Preview what would be processed without executing

## Testing

A test script is provided to verify the implementation:

```bash
python3 agent_bank/scripts/test_interview_builder.py
```

This tests the builder with a single agent using the first 3 questions.

## Workflow

1. **Extract Interview Script**
   ```bash
   python3 agent_bank/scripts/extract_interview_script.py
   ```

2. **Configure Builder** (optional)
   - Customize reflection triggers
   - Adjust save interval
   - Set manual reflection points

3. **Run Interview**
   ```python
   builder = InterviewMemoryBuilder(interview_script_path, config)
   result = builder.build_memory(agent_folder)
   ```

4. **Verify Results**
   - Check memory stream nodes
   - Verify reflections were created
   - Review agent responses for consistency

## Features

- ✅ Modular reflection trigger system
- ✅ Automatic Q&A memory saving
- ✅ Progress tracking and resume capability
- ✅ Batch processing support
- ✅ Interview-specific prompt template with "imaginary figure" framing
- ✅ Configurable save intervals
- ✅ Error handling and recovery

## Notes

- Agents must have `scratch.json` with demographic information
- Memory streams start empty and are populated during the interview
- Reflections are created as separate nodes with `node_type="reflection"`
- The interview uses the same format as the original human interviews for consistency

