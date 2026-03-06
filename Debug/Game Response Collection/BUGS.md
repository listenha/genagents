# Wavelength Game Response Collection — Bug Log

This folder tracks bugs found in the Wavelength game response collection pipeline (run Mar 2026, scale -100 to 100). Use this for debugging and future reference.

**Move this file to `Debug/Game Response Collection/BUGS.md` if desired.**

---

## Bug 1: Multiple attempts per agent when only one run was intended

### Type
**Behavior / data integrity**

### Observation
- Task was run once per population (single batch job), but many agents have **2 attempts** in `wavelength_responses.json`.
- Example: `Mistral-Nemo_agents/0057` has `attempt_id: 1` (all nulls) and `attempt_id: 2` (one parsed value, rest null).

### Root cause
1. **Pre-existing data**: Only agent `0000` had its `wavelength_responses.json` deleted before the "full" run. All other agents (0001–0199) still had previous trial data. The code **appends** a new attempt whenever `administer_game()` runs: `attempt_id = len(responses_data["attempts"]) + 1`.
2. **No "replace" mode**: The pipeline has no option to replace all attempts; it always appends.

### Diagnostic (Mistral)
- **agents_with_multiple_attempts**: 197 of 200
- **attempt_count_distribution**: 3 agents with 1 attempt, 197 with 2 attempts

### Location
- `Wavelength_Game/wavelength_response_builder.py`: `administer_game()`, lines ~412–432.

### Solution (recommended)
- Before a full re-collection: Delete or clear `wavelength_responses.json` for **all** agents (or clear the `attempts` array), not only 0000.
- Optional: Add a CLI flag (e.g. `--replace`) that overwrites with a single attempt.

---

## Bug 2: Enormous number of null responses (batch mode)

### Type
**Parsing / truncation**

### Observation
- Many headers have `"response": null` and `"raw_response": "ERROR: No response received"` or `"No reasoning provided"`.
- Example: 0057 attempt 2 has only 1 non-null response (60); headers 1–21 are null.

### Root cause
- **Batch mode** sends all 22 headers in one inference. Parser uses regex findall for `"Response": <number>`.
- If model output is **truncated** (max_tokens), only the first few headers appear → few numbers parsed → rest padded to None.
- `max_tokens` = max(2000, 22*300) = 6600; long reasoning per header can still exceed this.

### Diagnostic (Mistral)
- **average_null_rate_per_attempt**: 56.7% (so over half of all header slots are null across attempts).

### Location
- `genagents/modules/interaction.py`: `run_gpt_generate_numerical_resp()`, max_tokens.
- `Wavelength_Game/wavelength_response_builder.py`: `_administer_game_batch()` pads with None when `len(response_values) < len(header_order)`.

### Solution (recommended)
- **Prefer single-header mode**: Run without `--batch-by-header` so each header is one inference; avoids truncation.
- If keeping batch: Increase max_tokens and/or shorten prompt (e.g. minimal JSON).

---

## Bug 3: Responses appear only positive (no use of -100 to 0)

### Type
**Scale / prompt adherence**

### Observation
- Prompt updated to -100 to 100, but sampled files show only positive parsed values. Diagnostic confirms **no negative values** in the population.

### Diagnostic (Mistral)
- **any_negative_responses**: false
- **total_negative_values**: 0
- **global_min_response**: 0
- **global_max_response**: 90

### Possible causes
1. Parser: Already supports negative (`-?\d+\.?\d*` in llm_json_parser.py).
2. Model behavior: May favor 0–100 from habit; or truncation leaves only early (positive) headers.
3. Attempt 1 data (old 0–100 scale) has min 15–20, max 85–90; attempt 2 (new run) is mostly nulls, so the few parsed values might still be positive.

### Solution (recommended)
- Run `analyze_wavelength_collection.py` on each population; check `global_min_response` and `any_negative_responses`. If no negatives after fixing nulls, strengthen prompt (e.g. "You must use negative values when closer to left pole") and re-collect.

---

## Summary

| Bug | Main cause | Fix |
|-----|------------|-----|
| 1 Multi-attempt | Only 0000 cleared; append-only | Clear all agents before full run; optional --replace |
| 2 Nulls | Batch output truncation | Use single-header mode or raise max_tokens |
| 3 Positive-only | Confirmed: 0 min, 90 max; no negatives | Tighten prompt; re-check after #2 |

## Run diagnostics

From repo root:

```bash
python3 "Debug/Game Response Collection/analyze_wavelength_collection.py" --base agent_bank/populations/Mistral-Nemo_agents --out "Debug/Game Response Collection/mistral_diagnostic_report.json"
```

Interpret: attempt_count_distribution, average_null_rate_per_attempt, any_negative_responses, global_min_response.
