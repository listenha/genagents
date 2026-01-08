from pathlib import Path

# ============================================================================
# MODEL PROVIDER CONFIGURATION
# ============================================================================
# Choose between OpenAI API or local open-source models
# MODEL_PROVIDER = "openai"  # Options: "openai" or "local"
MODEL_PROVIDER = "local"  # Uncomment to use local models

# OpenAI Configuration (only needed if MODEL_PROVIDER == "openai")
OPENAI_API_KEY = "API_KEY"
KEY_OWNER = "NAME"

# ============================================================================
# MODEL SELECTION - Easy Switch Between Models
# ============================================================================
# Simply change MODEL_CHOICE to switch between available models

MODEL_CHOICE = "14b"  # Options: "7b", "14b", "32b", "32b-gptq", "32b-gptq-int4"

# Model paths configuration
MODEL_PATHS = {
    "7b": {
        "path": "/taiga/common_resources/models/Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        "name": "Qwen2.5-7B-Instruct"
    },
    "14b": {
        "path": "/taiga/common_resources/models/Qwen3-14B",
        "name": "Qwen3-14B"
    },
    "32b": {
        "path": "/taiga/common_resources/models/Qwen3-32B",
        "name": "Qwen3-32B"
    },
    "32b-gptq": {
        "path": "/taiga/common_resources/models/Qwen2.5-32B-Instruct-GPTQ-Int8",
        "name": "Qwen2.5-32B-Instruct-GPTQ-Int8"
    },
    "32b-gptq-int4": {
        "path": "/taiga/common_resources/models/Qwen3-32B-GPTQ-Int4",
        "name": "Qwen3-32B-GPTQ-Int4"
    }
}

# Auto-configure based on MODEL_CHOICE
_selected_model = MODEL_PATHS[MODEL_CHOICE]
LOCAL_MODEL_NAME = _selected_model["path"]
LLM_VERS = _selected_model["name"]

# Local Model Configuration (only needed if MODEL_PROVIDER == "local")
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # For embeddings
DEVICE = "cuda:1"  # Options: "cuda", "cpu", or "mps" (for Apple Silicon) - Using CUDA for GPU acceleration

# Optional: Hugging Face token (required for gated models)
HF_TOKEN = None  # Set to your HF token if needed: "hf_your_token_here"

DEBUG = False

MAX_CHUNK_SIZE = 4

# LLM version (for OpenAI) or model identifier (for local)
# LLM_VERS = "gpt-4o-mini"  # Used when MODEL_PROVIDER == "openai"
# LLM_VERS is now set automatically above based on MODEL_CHOICE

BASE_DIR = f"{Path(__file__).resolve().parent.parent}"

## To do: Are the following needed in the new structure? Ideally Populations_Dir is for the user to define.
POPULATIONS_DIR = f"{BASE_DIR}/agent_bank/populations" 
LLM_PROMPT_DIR = f"{BASE_DIR}/simulation_engine/prompt_template"

# ============================================================================
# MEMORY SYSTEM TUNABLE PARAMETERS
# ============================================================================
# These parameters control how the agent's memory system retrieves and uses
# memories during interactions. All parameters can be overridden in function
# calls, but these serve as global defaults for consistency.

# Memory Retrieval Count (n_count)
# Maximum number of memories to retrieve and include in agent description
# Higher values = more context but longer prompts and higher API costs
# Default: 120 memories per interaction
MEMORY_N_COUNT = 120

# Memory Hyperparameters (hp)
# Weights for the three-dimensional memory scoring system: [recency, relevance, importance]
# - recency_w: Weight for how recently memory was accessed (0-1)
# - relevance_w: Weight for semantic similarity to query (0-1) 
# - importance_w: Weight for memory importance score (0-1)
# Default: [0, 1, 0.5] means only relevance and importance matter (recency disabled)
# Example: [0.3, 0.5, 0.2] gives equal emphasis to all three dimensions
MEMORY_HP = [0, 1, 0.5]

# Recency Decay Factor
# Exponential decay rate for recency scoring (0-1)
# Higher values (closer to 1) = slower decay, older memories retain more weight
# Lower values = faster decay, only very recent memories matter
# Formula: recency_score = decay_factor ^ (max_timestep - last_retrieved)
# Default: 0.99 means ~1% decay per time step
MEMORY_RECENCY_DECAY = 0.99

# Reflection Count
# Number of reflection insights to generate when agent.reflect() is called
# Reflections are LLM-generated insights derived from existing memories
# Higher values = more diverse insights but more API calls and cost
# Default: 5 reflections per reflect() call
MEMORY_REFLECTION_COUNT = 1

# Reflection Retrieval Count
# Number of memories to retrieve when generating reflections
# These memories are used as context for the LLM to generate reflection insights
# Higher values = more context for reflections but longer prompts
# Default: 120 memories (same as MEMORY_N_COUNT)
MEMORY_RETRIEVAL_COUNT = 120

# Memory Time Step
# Default time step used for memory retrieval operations
# Used for recency calculations (determines "current time" for decay)
# Typically set to 0 or current simulation step
# Default: 0
MEMORY_TIME_STEP = 0

# Memory Filter Type
# Filter memories by node type during retrieval
# Options: "all" (both observations and reflections), "observation", "reflection"
# Default: "all" (include all memory types)
MEMORY_CURR_FILTER = "all"