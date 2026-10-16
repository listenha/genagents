"""
Central configuration for genagents.

This is the ONLY place where machine-specific information (model locations,
API keys, GPU selection, data location) enters the code base. Every value that
differs between servers is read from an environment variable, so the repo can
be cloned anywhere and run without editing code:

    cp .env.example .env          # fill in your values
    set -a; source .env; set +a   # export them into the shell
    python3 Surveys/run_survey.py --base agent_bank/populations/Qwen3-14B_agents --all

Everything else in this file (model registry, memory hyperparameters) is
experiment configuration shared by all machines.
"""

import os
from pathlib import Path


def _env(name, default=None):
  value = os.environ.get(name)
  return value if value not in (None, "") else default


# ============================================================================
# PATHS
# ============================================================================
# Repo-relative paths are derived from this file's location, so the working
# directory never matters.
BASE_DIR = str(Path(__file__).resolve().parent.parent)
LLM_PROMPT_DIR = f"{BASE_DIR}/simulation_engine/prompt_template"
POPULATIONS_DIR = _env("GENAGENTS_POPULATIONS_DIR",
                       f"{BASE_DIR}/agent_bank/populations")

# Directory holding locally downloaded model weights, one sub-folder per model
# (see download_models.py). If unset, or if a model's folder is missing, the
# Hugging Face hub id is used instead and weights are fetched into the HF cache
# (controlled by the standard HF_HOME variable).
MODELS_DIR = _env("GENAGENTS_MODELS_DIR")


# ============================================================================
# MODEL PROVIDER
# ============================================================================
# "local": Hugging Face transformers on this machine's GPUs.
# "openai": OpenAI API (requires OPENAI_API_KEY).
MODEL_PROVIDER = _env("GENAGENTS_PROVIDER", "local")

OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_MODEL = _env("GENAGENTS_OPENAI_MODEL", "gpt-5-mini")

# Required only for gated models (Llama, Gemma, Ministral).
HF_TOKEN = _env("HF_TOKEN")

# "cuda" auto-selects the GPU with the most free memory; "cuda:N", "cpu" and
# "mps" are also accepted. Use CUDA_VISIBLE_DEVICES to restrict visible GPUs.
DEVICE = _env("GENAGENTS_DEVICE", "cuda")

LOCAL_EMBEDDING_MODEL = _env("GENAGENTS_EMBEDDING_MODEL",
                             "sentence-transformers/all-MiniLM-L6-v2")

DEBUG = _env("GENAGENTS_DEBUG", "0") == "1"


# ============================================================================
# LOCAL MODEL REGISTRY
# ============================================================================
# Short key -> Hugging Face id and display name. Select one with the
# GENAGENTS_MODEL environment variable (or --model-choice on the patch
# scripts). The local folder name is the last component of the hub id, which is
# the layout produced by download_models.py.
#
# Population folders used in the TUX paper:
#   14b -> Qwen3-14B_agents, 7b -> Qwen2.5-7B_agents,
#   llama3.1-8b -> Llama-3.1-8B_agents, mistral-nemo-2407 -> Mistral-Nemo_agents,
#   gemma-3-12b-it -> gemma-3-12b-it_agents,
#   ministral-3-14b-instruct-2512 -> Ministral-3-14B-Instruct-2512_agents
MODEL_REGISTRY = {
  "7b": {
    "hf_id": "Qwen/Qwen2.5-7B-Instruct",
    "name": "Qwen2.5-7B-Instruct",
  },
  "14b": {
    "hf_id": "Qwen/Qwen3-14B",
    "name": "Qwen3-14B",
  },
  "llama3.1-8b": {
    "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
    "name": "Llama-3.1-8B-Instruct",
  },
  "mistral-nemo-2407": {
    "hf_id": "mistralai/Mistral-Nemo-Instruct-2407",
    "name": "Mistral-Nemo-Instruct-2407",
  },
  "gemma-3-12b-it": {
    "hf_id": "google/gemma-3-12b-it",
    "name": "google/gemma-3-12b-it",
  },
  "ministral-3-14b-instruct-2512": {
    "hf_id": "mistralai/Ministral-3-14B-Instruct-2512",
    "name": "mistralai/Ministral-3-14B-Instruct-2512",
  },
  "32b": {
    "hf_id": "Qwen/Qwen3-32B",
    "name": "Qwen3-32B",
  },
  "32b-gptq": {
    "hf_id": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
    "name": "Qwen2.5-32B-Instruct-GPTQ-Int8",
  },
}

MODEL_CHOICE = _env("GENAGENTS_MODEL", "14b")
if MODEL_CHOICE not in MODEL_REGISTRY:
  raise ValueError(f"Unknown GENAGENTS_MODEL '{MODEL_CHOICE}'. "
                   f"Valid choices: {list(MODEL_REGISTRY)}")


def resolve_local_model_path(choice):
  """Local weights folder if present under MODELS_DIR, else the HF hub id."""
  hf_id = MODEL_REGISTRY[choice]["hf_id"]
  if MODELS_DIR:
    local_path = Path(MODELS_DIR) / hf_id.split("/")[-1]
    if local_path.exists():
      return str(local_path)
  return hf_id


LOCAL_MODEL_NAME = resolve_local_model_path(MODEL_CHOICE)
LOCAL_MODEL_HF_ID = MODEL_REGISTRY[MODEL_CHOICE]["hf_id"]

# Model identifier passed through the agent code. For OpenAI this is the API
# model name; for local models it is a display name only.
if MODEL_PROVIDER == "openai":
  LLM_VERS = OPENAI_MODEL
else:
  LLM_VERS = MODEL_REGISTRY[MODEL_CHOICE]["name"]


# ============================================================================
# MEMORY SYSTEM PARAMETERS
# ============================================================================
# Global defaults for memory retrieval; all can be overridden per call.

# Maximum number of memories retrieved into the agent description per query.
MEMORY_N_COUNT = 120

# Weights for [recency, relevance, importance] in retrieval scoring.
# [0, 1, 0.5] disables recency.
MEMORY_HP = [0, 1, 0.5]

# Recency score = decay ^ (max_timestep - last_retrieved).
MEMORY_RECENCY_DECAY = 0.99

# Number of reflection nodes generated per agent.reflect() call.
MEMORY_REFLECTION_COUNT = 1

# Number of memories retrieved as context when generating reflections.
MEMORY_RETRIEVAL_COUNT = 120

# Default "current" time step for retrieval (used in recency calculation).
MEMORY_TIME_STEP = 0

# Node types considered during retrieval: "all", "observation", "reflection".
MEMORY_CURR_FILTER = "all"
