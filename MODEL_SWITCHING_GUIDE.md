# Guide: Switching Between Qwen Models

## Overview

This guide explains how to download and switch between different Qwen models in your codebase.

## Current Model Setup

Your current model is located at:
```
/srv/local/common_resources/models/Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

And configured in:
```
genagents/simulation_engine/settings.py
```

## Step 1: Download New Models

### Option A: Using the Download Script (Recommended)

1. **Activate your virtual environment** (if needed):
   ```bash
   cd /srv/local/common_resources/yueshen7/genagents
   source venv/bin/activate  # or your venv path
   ```

2. **Set Hugging Face token** (if required for gated models):
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

3. **Run the download script**:
   ```bash
   python3 download_models.py
   ```

   This will download:
   - `Qwen/Qwen2.5-32B-Instruct`
   - `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8` (if available)

   Models will be saved to: `/srv/local/common_resources/models/`

### Option B: Manual Download Using HuggingFace CLI

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download model (will use HF cache by default)
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir /srv/local/common_resources/models/Qwen2.5-32B-Instruct
```

### Option C: Using Python Directly

```python
from huggingface_hub import snapshot_download

# Download to custom directory
snapshot_download(
    repo_id="Qwen/Qwen2.5-32B-Instruct",
    local_dir="/srv/local/common_resources/models/Qwen2.5-32B-Instruct",
    local_dir_use_symlinks=False
)
```

## Step 2: Find the Snapshot Path

After downloading, find the snapshot directory:

```bash
ls /srv/local/common_resources/models/Qwen2.5-32B-Instruct/snapshots/
```

You'll see a directory with a hash like: `a09a35458c702b33eeacc393d103063234e8bc28`

The full path will be:
```
/srv/local/common_resources/models/Qwen2.5-32B-Instruct/snapshots/<hash>
```

## Step 3: Switch Models in settings.py

Edit `genagents/simulation_engine/settings.py`:

### For Local Path (Recommended - matches current setup):

```python
# For Qwen2.5-32B-Instruct (standard)
LOCAL_MODEL_NAME = "/srv/local/common_resources/models/Qwen2.5-32B-Instruct/snapshots/<hash>"
LLM_VERS = "Qwen2.5-32B-Instruct"

# OR for GPTQ quantized version (smaller, faster)
LOCAL_MODEL_NAME = "/srv/local/common_resources/models/Qwen2.5-32B-Instruct-GPTQ-Int8/snapshots/<hash>"
LLM_VERS = "Qwen2.5-32B-Instruct-GPTQ-Int8"
```

### For HuggingFace Model ID (Alternative):

If you want to use HuggingFace model IDs directly (downloads to HF cache):

```python
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"  # HuggingFace model ID
LLM_VERS = "Qwen2.5-32B-Instruct"
```

## Step 4: Verify Model Loading

Test that the model loads correctly:

```python
from simulation_engine.local_model_adapter import load_local_model

model, tokenizer = load_local_model()
print(f"Model loaded: {model.config.name_or_path}")
```

## Model Comparison

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| Qwen2.5-7B-Instruct | ~14GB | Fast | Good | Current default |
| Qwen2.5-32B-Instruct | ~60GB | Slower | Better | Higher quality needed |
| Qwen2.5-32B-Instruct-GPTQ-Int8 | ~18GB | Medium | Similar to 32B | Faster 32B variant |

## Notes on GPTQ Models

- GPTQ models are quantized (compressed) versions that use less memory
- They require special loading (may need `auto_gptq` package)
- Check the model card on HuggingFace for specific loading instructions
- Some GPTQ models are available from `TheBloke` organization on HuggingFace

## Troubleshooting

1. **Model not found**: Check the path and ensure the snapshot directory exists
2. **Out of memory**: 32B models need significant GPU memory (24GB+ VRAM recommended)
3. **Slow loading**: First load caches the model, subsequent loads are faster
4. **Authentication error**: Set `HF_TOKEN` in settings.py or environment variable

## Example: Quick Switch Script

Create a helper script to quickly switch models:

```python
# switch_model.py
import sys
from pathlib import Path

MODELS = {
    "7b": {
        "path": "/srv/local/common_resources/models/Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        "name": "Qwen2.5-7B-Instruct"
    },
    "32b": {
        "path": "/srv/local/common_resources/models/Qwen2.5-32B-Instruct/snapshots/<hash>",
        "name": "Qwen2.5-32B-Instruct"
    },
    "32b-gptq": {
        "path": "/srv/local/common_resources/models/Qwen2.5-32B-Instruct-GPTQ-Int8/snapshots/<hash>",
        "name": "Qwen2.5-32B-Instruct-GPTQ-Int8"
    }
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python switch_model.py <model_key>")
        print(f"Available models: {list(MODELS.keys())}")
        sys.exit(1)
    
    model_key = sys.argv[1]
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        sys.exit(1)
    
    # Read settings.py
    settings_path = Path(__file__).parent / "simulation_engine" / "settings.py"
    with open(settings_path, 'r') as f:
        content = f.read()
    
    # Update model settings
    model_config = MODELS[model_key]
    # ... (update logic here)
    
    print(f"Switched to: {model_config['name']}")
```

