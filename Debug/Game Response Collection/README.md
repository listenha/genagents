# Debug: Game Response Collection

This folder holds **debug assets for the Wavelength game response collection** pipeline: scripts, data, and documentation used to find and fix bugs (multiple attempts, null responses, scale -100 to 100 usage).

## Purpose

- **Manage the debug workflow**: Keep all debugging for "game response collection" in one place.
- **Document bugs**: Record bug type, cause, and solution for future reference (see [BUGS.md](BUGS.md)).

## Contents

| Item | Description |
|------|-------------|
| **BUGS.md** | Bug log: multiple attempts, null responses, positive-only scale. Root causes and recommended fixes. |
| **analyze_wavelength_collection.py** | Diagnostic script: scans `wavelength_responses.json` across a population and reports attempt counts, null rates, and response value distribution (min/max/negative count). |
| **mistral_diagnostic_report.json** | Output of the diagnostic script on Mistral-Nemo_agents (200 agents). |
| **README.md** | This file (move to `Game Response Collection/README.md` if keeping there). |

## Quick start

1. **Read the bug log**: [BUGS.md](BUGS.md) for the three issues (multi-attempt, nulls, scale).
2. **Run the diagnostic** (from repo root):
   ```bash
   python3 "Debug/Game Response Collection/analyze_wavelength_collection.py" --base agent_bank/populations/Mistral-Nemo_agents --out "Debug/Game Response Collection/mistral_diagnostic_report.json"
   ```
3. **Interpret output**: See "Run diagnostics" in BUGS.md for what each metric means.

## Relevant code paths

- **Builder / game flow**: `Wavelength_Game/wavelength_response_builder.py` (`administer_game`, `_administer_game_batch`, `_administer_game_single`).
- **Parsing**: `simulation_engine/llm_json_parser.py` (`extract_first_json_dict_numerical`).
- **Numerical response pipeline**: `genagents/modules/interaction.py` (`run_gpt_generate_numerical_resp`, max_tokens).
- **Prompts**: `simulation_engine/prompt_template/generative_agent/interaction/numerical_resp/wavelength_singular_v1.txt`, `wavelength_batch_v1.txt`.

## Example problematic file

- `agent_bank/populations/Mistral-Nemo_agents/0057/wavelength_responses.json`: two attempts; attempt 1 all nulls; attempt 2 only one non-null (60), rest null with "ERROR: No response received".
