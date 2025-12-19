# Quick Start: Downloading and Switching Models

## Step 1: Download Models

Run the download script (this will take a while - models are large!):

```bash
cd /srv/local/common_resources/yueshen7/genagents

# Activate venv if needed
source venv/bin/activate

# Set HF token if needed (for gated models)
export HF_TOKEN="hf_your_token_here"  # Optional

# Run download script
python3 download_models.py
```

This downloads models to: `/srv/local/common_resources/models/`

**Note**: 
- "Qwen3-32B" doesn't exist yet - the script uses `Qwen2.5-32B-Instruct`
- GPTQ model names may vary - check HuggingFace for exact name
- 32B models are ~60GB each - ensure sufficient disk space!

## Step 2: Find Snapshot Paths

After download completes, find the snapshot paths:

```bash
# For Qwen2.5-32B-Instruct
ls /srv/local/common_resources/models/Qwen2.5-32B-Instruct/snapshots/

# For GPTQ version (if downloaded)
ls /srv/local/common_resources/models/Qwen2.5-32B-Instruct-GPTQ-Int8/snapshots/
```

You'll see a directory with a hash - that's your snapshot path!

## Step 3: Switch Model in settings.py

Edit `genagents/simulation_engine/settings.py`:

```python
# Change this line (around line 16):
LOCAL_MODEL_NAME = "/srv/local/common_resources/models/Qwen2.5-32B-Instruct/snapshots/<YOUR_HASH_HERE>"

# And update this line (around line 29):
LLM_VERS = "Qwen2.5-32B-Instruct"
```

Replace `<YOUR_HASH_HERE>` with the actual hash from Step 2.

## Quick Model Switching

To switch between models, just change `LOCAL_MODEL_NAME` and `LLM_VERS` in `settings.py`:

**For 7B model (current):**
```python
LOCAL_MODEL_NAME = "/srv/local/common_resources/models/Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LLM_VERS = "Qwen2.5-7B-Instruct"
```

**For 32B model (new):**
```python
LOCAL_MODEL_NAME = "/srv/local/common_resources/models/Qwen2.5-32B-Instruct/snapshots/<hash>"
LLM_VERS = "Qwen2.5-32B-Instruct"
```

**For 32B GPTQ model (if available):**
```python
LOCAL_MODEL_NAME = "/srv/local/common_resources/models/Qwen2.5-32B-Instruct-GPTQ-Int8/snapshots/<hash>"
LLM_VERS = "Qwen2.5-32B-Instruct-GPTQ-Int8"
```

That's it! The code will automatically use the new model on next run.

