# Agent Profile Summary Module

This module generates holistic profile summaries for agents by synthesizing their demographic information (`scratch.json`) and interview memory nodes (`memory_stream/nodes.json`) into a single coherent paragraph.

## Overview

The module uses a prompt template to instruct an LLM to infer stable patterns in personality, values, emotional tone, and worldview from the agent's data. The generated summary is appended to the agent's `scratch.json` file as the `agent_profile_summary` field.

## Features

- **Batch Processing**: Process single agents, ranges, specific lists, or all agents
- **Skip Existing**: Automatically skips agents that already have a summary
- **Error Handling**: Logs errors and continues processing remaining agents
- **Logging**: Comprehensive logging to both console and log file
- **Local Model Support**: Works with local open-source models (Qwen, Llama, etc.)

## Requirements

- Python 3.7+
- Configured `settings.py` with model provider settings
- Agent folders with:
  - `scratch.json` (demographic/background info)
  - `memory_stream/nodes.json` (interview memory nodes)

## Usage

### Basic Examples

```bash
# Generate summary for a single agent
python3 Agent_Profile_Summary/run_profile_summary.py --agent agent_bank/populations/gss_agents/0000

# Generate summaries for range of agents (0000 to 0049)
python3 Agent_Profile_Summary/run_profile_summary.py --base agent_bank/populations/gss_agents --range 0000-0049

# Generate summaries for multiple specific agents
python3 Agent_Profile_Summary/run_profile_summary.py --base agent_bank/populations/gss_agents --agents 0000 0001 0002

# Generate summaries for all agents in directory
python3 Agent_Profile_Summary/run_profile_summary.py --base agent_bank/populations/gss_agents --all
```

### Advanced Options

```bash
# Use custom prompt template
python3 Agent_Profile_Summary/run_profile_summary.py --base agent_bank/populations/gss_agents --range 0000-0049 --template custom_prompt.txt

# Adjust max tokens for longer summaries
python3 Agent_Profile_Summary/run_profile_summary.py --base agent_bank/populations/gss_agents --range 0000-0049 --max-tokens 800

# Specify custom log file
python3 Agent_Profile_Summary/run_profile_summary.py --base agent_bank/populations/gss_agents --range 0000-0049 --log-file my_log.log

# Dry run (show what would be processed)
python3 Agent_Profile_Summary/run_profile_summary.py --base agent_bank/populations/gss_agents --range 0000-0049 --dry-run
```

## Command Line Arguments

### Required (one of):
- `--agent PATH`: Path to a single agent folder
- `--base PATH`: Base path to agent bank directory

### Optional (with --base):
- `--range START-END`: Range of agent IDs (e.g., `0000-0049`)
- `--agents ID1 ID2 ...`: Specific agent IDs to process
- `--all`: Process all agents in base directory

### Configuration:
- `--template PATH`: Path to prompt template file (default: `Agent_Profile_Summary/prompt.txt`)
- `--max-tokens N`: Maximum tokens for LLM response (default: 500)
- `--log-file PATH`: Path to log file (default: `logs/profile_summary_YYYYMMDD_HHMMSS.log`)

### Execution:
- `--dry-run`: Show what would be processed without actually generating summaries

## Output

### Summary Field

The generated summary is appended to `scratch.json` as:

```json
{
  "first_name": "John",
  "last_name": "Doe",
  ...
  "agent_profile_summary": "John is a 30-year-old software engineer with a background in..."
}
```

### Logging

- **Console**: Real-time progress and results
- **Log File**: Detailed log saved to `logs/profile_summary_YYYYMMDD_HHMMSS.log`
- **Error Tracking**: Failed agents are logged with error messages for easy re-processing

## Prompt Template

The default prompt template (`prompt.txt`) instructs the LLM to:

1. Synthesize demographic information and interview memories
2. Infer stable patterns in personality, values, and worldview
3. Write a single cohesive paragraph (120-160 words)
4. Follow a specific structure with placeholders for:
   - Name, age, life stage, role
   - Background (education/work/environment)
   - Key life contexts
   - Personality traits
   - Values
   - Recurring concerns/insights
   - Formative themes
   - Interaction style
   - Core worldview

The template uses placeholders:
- `{{PASTE SCRATCH.JSON CONTENT HERE}}` - Replaced with formatted scratch.json
- `{{PASTE NODES.JSON CONTENT HERE}}` - Replaced with formatted nodes.json

## Model Configuration

The module uses the same model configuration as other modules:

### Using Local Models

Set in `simulation_engine/settings.py`:

```python
MODEL_PROVIDER = "local"
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # or Qwen/Qwen2.5-14B-Instruct
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda"
LLM_VERS = "Qwen2.5-7B-Instruct"
```

### Using OpenAI Models

```python
MODEL_PROVIDER = "openai"
OPENAI_API_KEY = "your-api-key"
LLM_VERS = "gpt-4o-mini"  # or gpt-4o, gpt-4-turbo, etc.
```

## Error Handling

- **Missing Files**: If `scratch.json` or `nodes.json` is missing, the agent is skipped with an error logged
- **LLM Errors**: If LLM generation fails, the error is logged and processing continues
- **Existing Summaries**: Agents with non-empty `agent_profile_summary` are automatically skipped
- **JSON Errors**: Malformed JSON files are logged as errors

## Batch Processing Tips

1. **Start Small**: Test with a single agent or small range first
2. **Monitor Logs**: Check the log file for errors and re-run failed agents
3. **Resume Capability**: The script automatically skips agents with existing summaries, so you can safely re-run on the same range
4. **Token Limits**: Adjust `--max-tokens` if summaries are being truncated

## Example Workflow

```bash
# 1. Test with one agent
python3 Agent_Profile_Summary/run_profile_summary.py --agent agent_bank/populations/gss_agents/0000

# 2. Check the output in scratch.json
cat agent_bank/populations/gss_agents/0000/scratch.json | grep -A 5 agent_profile_summary

# 3. Run on a range
python3 Agent_Profile_Summary/run_profile_summary.py --base agent_bank/populations/gss_agents --range 0000-0199

# 4. Check the log for any errors
tail -f logs/profile_summary_*.log

# 5. Re-run to process any failed agents (skips successful ones)
python3 Agent_Profile_Summary/run_profile_summary.py --base agent_bank/populations/gss_agents --range 0000-0199
```

## Troubleshooting

**Issue**: "GENERATION ERROR" in logs
- **Solution**: Check model configuration in `settings.py` and ensure the model is accessible

**Issue**: Summaries are too short
- **Solution**: Increase `--max-tokens` (e.g., `--max-tokens 800`)

**Issue**: Some agents skipped unexpectedly
- **Solution**: Check if they already have `agent_profile_summary` in their `scratch.json`

**Issue**: JSON formatting errors
- **Solution**: Ensure `scratch.json` and `nodes.json` are valid JSON files

