# Wavelength: Causes of Null/Dirty Responses and Proposed Solutions

## 1. Batch mode (all headers in one inference)

**Cause:** When `batch_by_header=True` (or `--batch-by-header`), all 22 headers are sent in a single prompt. The prompt includes:
- Full agent description (scratch + retrieved memories, up to 120 memories)
- All 22 spectrum/clue question blocks

This can:
- **Exceed model context length** → input truncated or model fails
- **Exceed output token limit** → model response is cut off before all 22 answers are written → incomplete JSON → parser returns fewer than 22 values → remaining slots become `null`

**Solution:**
- **Use one header per inference (default).** Do not pass `--batch-by-header`. The runner uses `batch_by_header: False` by default; keep it that way for reliable collection.
- If you must use batch, reduce the number of headers per call (e.g. 5–8) and run multiple batches per agent (would require a code change to support “chunked batch” mode).

---

## 2. Parser only accepts unquoted numerical Response

**Cause:** In `simulation_engine/llm_json_parser.py`, `extract_first_json_dict_numerical` uses:
```python
response_pattern = re.compile(r'"Response":\s*(\d+\.?\d*)')
```
This matches `"Response": 50` or `"Response": 50.5` but **not** `"Response": "50"` (string). Some models output JSON with quoted numbers; those responses are then missed and recorded as null.

**Solution:**
- Extend the regex (or add a fallback) to also match `"Response": "50"` and parse the captured string as int/float. For example, add a second pattern for quoted numbers and merge results, or use a single pattern that captures either quoted or unquoted numbers.

---

## 3. Model output format mismatch

**Cause:** The Wavelength prompt asks for nested JSON like `{"1": {"Q": "...", "Reasoning": "...", "Response": 50}, "2": {...}}`. If the model:
- Uses different key names (e.g. `"response"` lowercase)
- Outputs a list instead of keyed object
- Returns truncated or malformed JSON

then the parser (which uses regex findall for `"Reasoning"` and `"Response"`) may get wrong order or no matches → nulls.

**Solution:**
- Prefer **singular (one header per inference)** so the expected output is a single `{"1": {..., "Response": N}}` and parsing is more robust.
- Optionally add a fallback parser that tries to extract any number in a known range (0–100) when the primary parser returns empty.

---

## 4. API/run errors (timeout, OOM, network)

**Cause:** Any runtime error (timeout, out-of-memory, network) causes the try/except in `_administer_game_single` or `_administer_game_batch` to set `response: null` and store an error message in `raw_response`.

**Solution:**
- Monitor logs for "ERROR: No response received" or "ERROR: ...".
- Re-run only failed agents (or re-run with smaller batch / one-at-a-time) after fixing environment (e.g. more memory, longer timeout).

---

## 5. Max output tokens too low (batch)

**Cause:** In `genagents/modules/interaction.py`, for batch numerical_resp:
```python
calculated_max_tokens = max(2000, num_questions * 300)
```
For 22 questions that is 6600 tokens. Each Wavelength answer can be long (Range Interpretation + Reasoning + Response). If the model needs more than 6600 tokens to output all 22, the response is truncated → incomplete JSON → nulls for later headers.

**Solution:**
- Use **one header per inference** (no batch) so each response is short and within default limits.
- If you keep batch, increase max_tokens for the batch call (e.g. 500–600 per question) or reduce headers per batch.

---

## Recommended configuration for clean data

- **Do not use** `--batch-by-header`. Use default one-header-per-inference.
- Re-run the game with `batch_by_header: False` (default) so every agent gets 22 separate inferences and responses are recorded one-by-one without truncation or batch parsing issues.
