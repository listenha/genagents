#!/usr/bin/env python3
"""
Download model weights into GENAGENTS_MODELS_DIR.

Each model is stored in <GENAGENTS_MODELS_DIR>/<last part of hub id>, which is
where simulation_engine/settings.py looks for it. Models are chosen by their
MODEL_REGISTRY key; with no arguments, the four model families used in the TUX
paper are downloaded.

Usage:
  export GENAGENTS_MODELS_DIR=/path/to/models
  export HF_TOKEN=hf_...            # needed for gated models (Llama, Gemma, Ministral)
  python3 download_models.py                    # paper models
  python3 download_models.py gemma-3-12b-it 32b # specific registry keys
"""

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from simulation_engine.settings import MODEL_REGISTRY, MODELS_DIR, HF_TOKEN

PAPER_MODELS = ["14b", "7b", "llama3.1-8b", "mistral-nemo-2407"]


def download_model(hf_id: str, base_dir: Path) -> bool:
  target_dir = base_dir / hf_id.split("/")[-1]
  print(f"\n{'=' * 60}\nDownloading {hf_id}\n  -> {target_dir}\n{'=' * 60}")
  try:
    snapshot_download(repo_id=hf_id, local_dir=str(target_dir), token=HF_TOKEN)
  except Exception as e:
    print(f"✗ Error downloading {hf_id}: {e}")
    print("  Check disk space, network access, and HF_TOKEN for gated models.")
    return False
  print(f"✓ {hf_id} downloaded")
  return True


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("models", nargs="*", metavar="MODEL",
                      help=f"Registry keys to download: {', '.join(MODEL_REGISTRY)} "
                           f"(default: {' '.join(PAPER_MODELS)})")
  args = parser.parse_args()
  models = args.models or PAPER_MODELS
  unknown = [key for key in models if key not in MODEL_REGISTRY]
  if unknown:
    parser.error(f"unknown model key(s): {', '.join(unknown)}")

  if not MODELS_DIR:
    sys.exit("GENAGENTS_MODELS_DIR is not set. Point it at the directory that should hold model weights.")
  base_dir = Path(MODELS_DIR)
  base_dir.mkdir(parents=True, exist_ok=True)
  print(f"Models directory: {base_dir}")
  print(f"Available disk space: {shutil.disk_usage(base_dir).free / 1024**3:.1f} GB")

  failed = [key for key in models
            if not download_model(MODEL_REGISTRY[key]["hf_id"], base_dir)]

  print(f"\nDownloaded {len(models) - len(failed)}/{len(models)} models.")
  if failed:
    print(f"Failed: {', '.join(failed)}")
    sys.exit(1)


if __name__ == "__main__":
  main()
