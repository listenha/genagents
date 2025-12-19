#!/usr/bin/env python3
"""
Script to download Qwen models to the local models directory.

This script downloads models to: /srv/local/common_resources/models/
which matches the structure used for Qwen2.5-7B-Instruct
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# Models to download
# Note: "Qwen3-32B" doesn't exist yet - using Qwen2.5-32B-Instruct instead
# If you need a different model, update these names
MODELS_TO_DOWNLOAD = [
    "Qwen/Qwen3-14B",
    "JunHowie/Qwen3-32B-GPTQ-Int4"
    # "Qwen/Qwen2.5-32B-Instruct",  # Standard 32B model
    # "Qwen/Qwen3-32B",
    # "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
    # GPTQ models may have different names - check HuggingFace
    # Common: "TheBloke/Qwen2.5-32B-Instruct-GPTQ" or similar
    # "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",  # Uncomment if this exact name exists
]

# Base directory for models
BASE_MODELS_DIR = Path("/srv/local/common_resources/models")

# Optional: Hugging Face token (required for gated models)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

def download_model(model_name: str, base_dir: Path, token: str = None):
    """
    Download a model using snapshot_download to maintain the proper structure.
    
    Args:
        model_name: Hugging Face model identifier (e.g., "Qwen/Qwen2.5-32B-Instruct")
        base_dir: Base directory where models will be stored
        token: Optional Hugging Face token for gated models
    """
    print(f"\n{'='*60}")
    print(f"Downloading model: {model_name}")
    print(f"{'='*60}")
    
    # Extract model folder name (e.g., "Qwen2.5-32B-Instruct" from "Qwen/Qwen2.5-32B-Instruct")
    model_folder_name = model_name.split('/')[-1]
    target_dir = base_dir / model_folder_name
    
    # Create base directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use snapshot_download to download the model
        # This creates the proper snapshots/ directory structure
        auth_kwargs = {}
        if token:
            auth_kwargs['token'] = token
            print(f"Using Hugging Face token for authentication")
        
        print(f"Target directory: {target_dir}")
        print(f"This may take a while depending on model size and internet speed...")
        
        snapshot_download(
            repo_id=model_name,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,  # Copy files instead of symlinks
            token=token if token else None,
            resume_download=True,  # Resume if interrupted
        )
        
        # Find the snapshot directory (HuggingFace creates snapshots/<hash>/ structure)
        snapshot_dirs = list(target_dir.glob("snapshots/*"))
        if snapshot_dirs:
            snapshot_path = snapshot_dirs[0]
            print(f"\n✓ Model downloaded successfully!")
            print(f"  Model path: {snapshot_path}")
            print(f"  Use this path in settings.py: {snapshot_path}")
        else:
            # If no snapshots dir, the model might be directly in target_dir
            print(f"\n✓ Model downloaded successfully!")
            print(f"  Model path: {target_dir}")
            print(f"  Use this path in settings.py: {target_dir}")
            
    except Exception as e:
        print(f"\n✗ Error downloading {model_name}: {e}")
        print(f"  Make sure you have:")
        print(f"  1. Sufficient disk space (32B models are large!)")
        print(f"  2. Internet connection")
        print(f"  3. Hugging Face token if model is gated (set HF_TOKEN environment variable)")
        return False
    
    return True

def main():
    print("Qwen Model Download Script")
    print("=" * 60)
    print(f"Models will be downloaded to: {BASE_MODELS_DIR}")
    print(f"Current models directory exists: {BASE_MODELS_DIR.exists()}")
    
    # Check available space (rough estimate)
    import shutil
    if BASE_MODELS_DIR.exists():
        stat = shutil.disk_usage(BASE_MODELS_DIR)
        free_gb = stat.free / (1024**3)
        print(f"Available disk space: {free_gb:.2f} GB")
        if free_gb < 100:
            print(f"  WARNING: 32B models require ~60-80GB each. Make sure you have enough space!")
    
    # Download each model
    successful = []
    failed = []
    
    for model_name in MODELS_TO_DOWNLOAD:
        if download_model(model_name, BASE_MODELS_DIR, HF_TOKEN):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    print(f"Successfully downloaded: {len(successful)}/{len(MODELS_TO_DOWNLOAD)}")
    for model in successful:
        print(f"  ✓ {model}")
    
    if failed:
        print(f"\nFailed to download: {len(failed)}/{len(MODELS_TO_DOWNLOAD)}")
        for model in failed:
            print(f"  ✗ {model}")
    
    if successful:
        print(f"\n{'='*60}")
        print("Next Steps:")
        print(f"{'='*60}")
        print("1. Find the snapshot path for each downloaded model:")
        for model in successful:
            model_folder = model.split('/')[-1]
            model_path = BASE_MODELS_DIR / model_folder
            snapshot_dirs = list(model_path.glob("snapshots/*"))
            if snapshot_dirs:
                print(f"   {model}: {snapshot_dirs[0]}")
            else:
                print(f"   {model}: {model_path}")
        print("\n2. Update simulation_engine/settings.py:")
        print("   LOCAL_MODEL_NAME = \"/path/to/model/snapshots/<hash>\"")
        print("   LLM_VERS = \"ModelName\"")

if __name__ == "__main__":
    main()

