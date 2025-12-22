# genagents: Generative Agent Simulations of 1,000 People

![Cover Image](static_dir/cover3.png)

## Overview

This project introduces a novel agent architecture that simulates the attitudes and behaviors of real individuals by applying large language models (LLMs) to qualitative interviews about their lives. These agents replicate participants' responses on various social science measures, providing a foundation for new tools to investigate individual and collective behavior.

This codebase offers two main components:

1. **Codebase for Creating and Interacting with Generative Agents**: Tools to build new agents based on your own data and interact with them. Query agents with surveys, experiments, and other stimuli to study their responses.
2. **Demographic Agent Banks**: A bank of over 3,000 agents created using demographic information from the General Social Survey (GSS) as a starting point to explore the codebase. *Note: The names and addresses are fictional.*

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Creating Agents](#creating-agents)
  - [Interacting with Agents](#interacting-with-agents)
  - [Memory and Reflection](#memory-and-reflection)
  - [Saving and Loading Agents](#saving-and-loading-agents)
- [Model Configuration](#model-configuration)
  - [Using OpenAI Models](#using-openai-models)
  - [Using Open Source Models](#using-open-source-models)
  - [Switching Between Models](#switching-between-models)
- [Agent Interaction Modules](#agent-interaction-modules)
  - [Interview Module](#interview-module)
  - [Surveys Module](#surveys-module)
- [Agent Memory System](#agent-memory-system)
- [Sample Agent](#sample-agent)
- [Agent Bank Access](#agent-bank-access)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Contact](#contact)

---

## Installation

### Requirements

- Python 3.7 or higher
- An OpenAI API key (for OpenAI models) OR local model setup (for open-source models)

### Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

**For Open Source Models** (optional):
```bash
pip install transformers torch sentence-transformers
```

**Note:** For GPU acceleration with open-source models, install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
```

---

## Configuration

Create a `settings.py` file in the `simulation_engine` folder (where `example-settings.py` is located). Place the following content in `settings.py`:

### Basic Configuration (OpenAI)

```python
from pathlib import Path

# Model Provider Configuration
MODEL_PROVIDER = "openai"  # Options: "openai" or "local"

# OpenAI Configuration
OPENAI_API_KEY = "YOUR_API_KEY"
KEY_OWNER = "YOUR_NAME"

# Debug mode
DEBUG = False
MAX_CHUNK_SIZE = 4

# LLM version
LLM_VERS = "gpt-4o-mini"

BASE_DIR = f"{Path(__file__).resolve().parent.parent}"
POPULATIONS_DIR = f"{BASE_DIR}/agent_bank/populations"
LLM_PROMPT_DIR = f"{BASE_DIR}/simulation_engine/prompt_template"
```

### Configuration for Open Source Models

```python
from pathlib import Path

# Model Provider Configuration
MODEL_PROVIDER = "local"  # Use local models

# Local Model Configuration
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Hugging Face model ID
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # For embeddings
DEVICE = "cuda"  # Options: "cuda", "cpu", or "mps" (for Apple Silicon)

# Optional: Hugging Face token (required for gated models)
HF_TOKEN = None  # Set to your HF token if needed: "hf_your_token_here"

# Debug mode
DEBUG = False
MAX_CHUNK_SIZE = 4

# LLM version (used as identifier)
LLM_VERS = "Qwen2.5-7B-Instruct"

BASE_DIR = f"{Path(__file__).resolve().parent.parent}"
POPULATIONS_DIR = f"{BASE_DIR}/agent_bank/populations"
LLM_PROMPT_DIR = f"{BASE_DIR}/simulation_engine/prompt_template"
```

Replace `"YOUR_API_KEY"` with your actual OpenAI API key and `"YOUR_NAME"` with your name.

---

## Repository Structure

```
genagents/
├── genagents/                          # Core module for creating and interacting with agents
│   ├── genagents.py                    # Main GenerativeAgent class
│   └── modules/                         # Submodules
│       ├── interaction.py               # Handles agent interactions and responses
│       └── memory_stream.py            # Manages agent memory and reflections
├── simulation_engine/                  # Settings and global methods
│   ├── settings.py                     # Configuration settings (create this)
│   ├── example-settings.py             # Example configuration
│   ├── global_methods.py               # Helper functions
│   ├── gpt_structure.py                # Functions for interacting with LLMs
│   ├── llm_json_parser.py             # Parses JSON outputs from language models
│   └── prompt_template/                # All LLM prompts used in this project
├── agent_bank/                         # Directory for storing agent data
│   └── populations/                   # Contains pre-generated agents
│       ├── gss_agents/                 # Demographic agent data based on 
├── Interview/                          # Interview module (see Interview/README.md)
│   ├── README.md                       # Interview module documentation
│   ├── run_interview.py                 # CLI for running interviews
│   └── ...
├── Surveys/                             # Surveys module (see Surveys/README.md)
│   ├── README.md                       # Surveys module documentation
│   ├── run_survey.py                   # CLI for running surveys
│   └── ...
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

---

## Quick Start

### 1. Create a New Agent

```python
from genagents.genagents import GenerativeAgent

# Initialize a new agent
agent = GenerativeAgent()

# Update the agent's scratchpad with personal information
agent.update_scratch({
    "first_name": "John",
    "last_name": "Doe",
    "age": 30,
    "occupation": "Software Engineer",
    "interests": ["reading", "hiking", "coding"]
})
```

### 2. Interact with the Agent

```python
# Categorical questions (multiple choice)
questions = {
    "Do you enjoy outdoor activities?": ["Yes", "No", "Sometimes"]
}
response = agent.categorical_resp(questions)
print(response["responses"])

# Numerical questions (scale)
questions = {
    "On a scale of 1 to 10, how much do you enjoy coding?": [1, 10]
}
response = agent.numerical_resp(questions, float_resp=False)
print(response["responses"])

# Open-ended questions
dialogue = [
    ("Interviewer", "Tell me about your favorite hobby."),
]
response = agent.utterance(dialogue)
print(response)
```

### 3. Save the Agent

```python
agent.save("path/to/save_directory")
```

---

## Core Concepts

### Creating Agents

#### Initialize a New Agent

```python
from genagents.genagents import GenerativeAgent

# Create a new agent
agent = GenerativeAgent()

# The agent now has:
# - A unique ID (automatically generated)
# - An empty scratchpad (personal information)
# - An empty memory stream
```

#### Populate Agent's Personal Information

The "scratchpad" contains the agent's core identity and characteristics:

```python
agent.update_scratch({
    "first_name": "John",
    "last_name": "Doe",
    "age": 30,
    "sex": "Male",
    "ethnicity": "Caucasian",
    "occupation": "Software Engineer",
    "education": "Bachelor's degree",
    "political_views": "Moderate",
    "religion": "None",
    "interests": ["reading", "hiking", "coding"],
    "personality_traits": "Introverted, analytical, detail-oriented",
    "life_experiences": "Grew up in a suburban area, moved to city for work",
    # Add any other relevant personal information
})
```

**Key Points:**
- The scratchpad is a dictionary that can contain any key-value pairs
- Common fields include: demographics, personality traits, life experiences, values, etc.
- This information is used by the LLM to generate contextually appropriate responses

#### Add Initial Memories (Optional but Recommended)

Memories help the agent maintain consistency and context across interactions:

```python
# Add memories about the agent's experiences
agent.remember("Graduated from college with a degree in Computer Science", time_step=1)
agent.remember("Started working as a software engineer at a tech company", time_step=2)
agent.remember("Enjoys solving complex programming problems", time_step=3)
agent.remember("Prefers working independently but values team collaboration", time_step=4)
```

**Note:** `time_step` represents the chronological order of events. Use sequential integers starting from 1.

#### Load an Existing Agent

```python
from genagents.genagents import GenerativeAgent

# Load an existing agent from a directory
agent = GenerativeAgent(agent_folder="agent_bank/populations/single_agent")

# Verify the agent loaded correctly
print(f"Agent name: {agent.get_fullname()}")
print(f"Agent description: {agent.get_self_description()}")
```

**Agent Folder Structure:**
A saved agent folder should contain:
- `scratch.json` - Personal information
- `memory_stream/nodes.json` - Memory nodes
- `memory_stream/embeddings.json` - Memory embeddings
- `meta.json` - Agent metadata

### Interacting with Agents

The agent can answer three types of questions:

1. **Categorical** - Multiple choice questions
2. **Numerical** - Scale/rating questions
3. **Open-ended** - Free-form text responses

#### Categorical Responses (Multiple Choice Surveys)

```python
# Define questions as a dictionary
# Key: question text
# Value: list of possible answer options
questions = {
    "Do you enjoy outdoor activities?": ["Yes", "No", "Sometimes"],
    "What is your preferred work environment?": ["Remote", "Office", "Hybrid"],
    "How do you prefer to spend your weekends?": ["Relaxing at home", "Socializing", "Pursuing hobbies", "Working"]
}

# Get agent's responses
response = agent.categorical_resp(questions)

# Response structure:
# {
#     "responses": ["Yes", "Hybrid", "Pursuing hobbies"],  # List of answers
#     "reasonings": ["reasoning for Q1", "reasoning for Q2", ...]  # Reasoning for each
# }

print("Responses:", response["responses"])
print("Reasonings:", response["reasonings"])
```

#### Numerical Responses (Scale Questions)

```python
# Define questions with numerical ranges
# Key: question text
# Value: [min_value, max_value] - the range of possible answers
questions = {
    "On a scale of 1 to 10, how much do you enjoy coding?": [1, 10],
    "Rate your work-life balance from 0 to 100": [0, 100],
    "How satisfied are you with your current salary? (1-5 scale)": [1, 5]
}

# Get responses (integer values)
response = agent.numerical_resp(questions, float_resp=False)

# Response structure:
# {
#     "responses": [8, 65, 3],  # List of numerical answers
#     "reasonings": ["reasoning for Q1", "reasoning for Q2", ...]
# }

print("Responses:", response["responses"])

# For float responses (decimal values)
questions_float = {
    "Rate your confidence level (0.0-1.0)": [0.0, 1.0]
}
response_float = agent.numerical_resp(questions_float, float_resp=True)
# Returns: {"responses": [0.75], ...}  # Float
```

#### Open-Ended Questions (Behavioral Choice-Making)

```python
# Create a dialogue format
# Each entry is a tuple: (speaker_name, message)
dialogue = [
    ("Interviewer", "Tell me about your favorite hobby."),
]

# Get agent's response
response = agent.utterance(dialogue)
print(f"Agent: {response}")

# Multi-turn conversation
conversation = []

# Question 1
conversation.append(("Interviewer", "What are your career goals?"))
response1 = agent.utterance(conversation)
conversation.append((agent.get_fullname(), response1))
print(f"Interviewer: What are your career goals?")
print(f"{agent.get_fullname()}: {response1}\n")

# Question 2 (with context from previous exchange)
conversation.append(("Interviewer", "How do you plan to achieve those goals?"))
response2 = agent.utterance(conversation)
conversation.append((agent.get_fullname(), response2))
print(f"Interviewer: How do you plan to achieve those goals?")
print(f"{agent.get_fullname()}: {response2}\n")
```

### Memory and Reflection

#### Adding Memories

Memories help agents maintain consistency and learn from interactions:

```python
# Add a memory about an interaction
agent.remember("Participated in a survey about work preferences", time_step=5)
agent.remember("Expressed preference for remote work", time_step=6)
agent.remember("Indicated high satisfaction with current role", time_step=7)
```

#### Reflection

Reflections allow agents to form insights from multiple memories:

```python
# Trigger reflection on a topic
# The agent will review relevant memories and generate insights
agent.reflect(anchor="work preferences", time_step=8)

# Reflection parameters:
# - anchor: Topic or keyword to reflect on
# - reflection_count: Number of insights to generate (default: 5)
# - retrieval_count: Number of memories to consider (default: 120)
# - time_step: Current time step

# Custom reflection
agent.reflect(
    anchor="career goals and motivations",
    reflection_count=3,  # Generate 3 insights
    retrieval_count=50,  # Consider top 50 relevant memories
    time_step=9
)
```

**How Reflection Works:**
1. The agent retrieves memories relevant to the anchor topic
2. The LLM analyzes these memories
3. New reflection nodes are created with insights
4. These reflections influence future responses

**How Memory Retrieval Works:**
- Memory retrieval happens **automatically** in every interaction
- Memories are **semantically selected** based on query relevance
- Selection uses **three-dimensional scoring** (recency, relevance, importance)
- Default: **120 memories** per interaction
- Format: Scratchpad + "==\n" + "Other observations..." + memory contents

See [Agent Memory System](#agent-memory-system) for detailed information.

### Saving and Loading Agents

#### Saving an Agent

```python
# Save agent to a directory
save_directory = "agent_bank/populations/my_agent"
agent.save(save_directory)

# The agent will be saved with:
# - scratch.json (personal information)
# - memory_stream/nodes.json (all memories and reflections)
# - memory_stream/embeddings.json (memory embeddings)
# - meta.json (agent metadata)
```

#### Loading a Saved Agent

```python
# Load the saved agent
agent = GenerativeAgent(agent_folder="agent_bank/populations/my_agent")

# Continue interacting
response = agent.categorical_resp({"Favorite color?": ["Red", "Blue", "Green"]})
```

---

## Model Configuration

### Using OpenAI Models

This is the default configuration. Set `MODEL_PROVIDER = "openai"` in `settings.py` and provide your OpenAI API key.

**Supported Models:**
- `gpt-4o-mini` (default, cost-effective)
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-3.5-turbo`
- `o1-preview` (special handling)

### Using Open Source Models

The genagents framework can be configured to use open-source models (like Qwen2.5-7B-Instruct) instead of OpenAI's GPT models.

#### Setup Steps

1. **Install Additional Dependencies:**
   ```bash
   pip install transformers torch sentence-transformers
   ```

2. **Update Settings Configuration:**
   ```python
   MODEL_PROVIDER = "local"
   LOCAL_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
   LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   DEVICE = "cuda"  # or "cpu" or "mps" (for Apple Silicon)
   HF_TOKEN = None  # Set if needed for gated models
   ```

3. **Create Local Model Adapter:**
   Create `simulation_engine/local_model_adapter.py` (see `USER_GUIDE.md` for full implementation details).

4. **Modify gpt_structure.py:**
   Add routing logic to use local models when `MODEL_PROVIDER == "local"`.

**Supported Models:**
- ✅ `Qwen/Qwen2.5-7B-Instruct`
- ✅ `Qwen/Qwen2.5-14B-Instruct`
- ✅ `meta-llama/Llama-3.1-8B-Instruct`
- ✅ `meta-llama/Llama-3.1-70B-Instruct`

**Embedding Models:**
- ✅ `sentence-transformers/all-MiniLM-L6-v2` (default, fast, 384-dim)
- ✅ `sentence-transformers/all-mpnet-base-v2` (slower, better quality, 768-dim)

**Note:** Once configured, usage remains the same! The agent code doesn't need to change.

### Switching Between Models

#### Quick Model Switching

To switch between models, change `LOCAL_MODEL_NAME` and `LLM_VERS` in `settings.py`:

**For 7B model:**
```python
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_VERS = "Qwen2.5-7B-Instruct"
```

**For 32B model:**
```python
LOCAL_MODEL_NAME = "/path/to/Qwen2.5-32B-Instruct/snapshots/<hash>"
LLM_VERS = "Qwen2.5-32B-Instruct"
```

#### Downloading Models

Use the download script to get models:

```bash
python3 download_models.py
```

Models are downloaded to `/srv/local/common_resources/models/` (or configured path).

#### Finding Snapshot Paths

After download, find the snapshot path:

```bash
ls /srv/local/common_resources/models/Qwen2.5-32B-Instruct/snapshots/
```

Use the hash directory as your `LOCAL_MODEL_NAME` path.

**Model Comparison:**

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| Qwen2.5-7B-Instruct | ~14GB | Fast | Good | Current default |
| Qwen2.5-32B-Instruct | ~60GB | Slower | Better | Higher quality needed |
| Qwen2.5-32B-Instruct-GPTQ | ~18GB | Medium | Similar to 32B | Faster 32B variant |

---

## Agent Interaction Modules

The codebase includes specialized modules for different types of agent interactions:

### Interview Module

**Location**: `Interview/`

The Interview module conducts semi-structured interviews with agents to build rich memory streams. Each question-answer pair is saved as an observation memory, and reflections are triggered based on configurable strategies.

**Documentation**: See [`Interview/README.md`](Interview/README.md) for complete documentation.

**Quick Start:**
```bash
# Run interview on a single agent
python3 Interview/run_interview.py --agent agent_bank/populations/gss_agents/0000

# Run on range of agents
python3 Interview/run_interview.py --base agent_bank/populations/gss_agents --range 0000-0049
```

**Features:**
- Modular reflection trigger system
- Automatic Q&A memory saving
- Progress tracking and resume capability
- Batch processing support
- Interview-specific prompt template

### Surveys Module

**Location**: `Surveys/`

The Surveys module administers standardized psychological surveys (PRE-TASK SURVEY) to agents, collects responses, and performs statistical analysis on consistency and distribution patterns.

**Documentation**: See [`Surveys/README.md`](Surveys/README.md) for complete documentation.

**Quick Start:**
```bash
# Run survey on a single agent
python3 Surveys/run_survey.py --agent agent_bank/populations/gss_agents/0000

# Run on range of agents
python3 Surveys/run_survey.py --base agent_bank/populations/gss_agents --range 0000-0049

# Analyze consistency
python3 Surveys/analyze_consistency.py --base agent_bank/populations/gss_agents --range 0000-0049 --attempts 1-10

# Analyze distribution
python3 Surveys/analyze_distribution.py --base agent_bank/populations/gss_agents --range 0000-0049 --attempts 1-10
```

**Features:**
- Survey administration with multiple attempts
- Consistency analysis (ICC, Cronbach's Alpha, etc.)
- Distribution analysis (box plots, violin plots, scatter plots)
- Response management (archive, translate)
- Interactive HTML reports

---

## Agent Memory System

The agent memory system is a core component that enables agents to maintain consistency and context across interactions.

### Memory Node Types

There are exactly **2 node types**:

1. **`"observation"`** - Direct memories/experiences
   - Created via `agent.remember()`
   - User-provided content
   - `pointer_id = None`

2. **`"reflection"`** - LLM-generated insights derived from observations
   - Created via `agent.reflect()`
   - LLM-generated content
   - `pointer_id = [list of source node IDs]`

### Memory Storage

**Before `agent.save()`:**
- All memory data is stored **in-memory** as Python objects
- `self.memory_stream.seq_nodes` - List of ConceptNode objects (in RAM)
- `self.memory_stream.id_to_node` - Dictionary for node lookup (in RAM)
- `self.memory_stream.embeddings` - Dictionary of content→embedding mappings (in RAM)

**After `agent.save()`:**
- Data is serialized to JSON files on disk
- `memory_stream/nodes.json` - All memory nodes
- `memory_stream/embeddings.json` - Memory embeddings

### Memory Retrieval

Memory retrieval happens **automatically** in every interaction:

1. **Query Formation**: Current question/topic becomes the "anchor"
2. **Semantic Search**: Embeddings find relevant memories
3. **Scoring**: Combines three dimensions:
   - **Recency**: Exponential decay based on `last_retrieved` time step
   - **Relevance**: Cosine similarity between query and memory embeddings
   - **Importance**: Pre-computed LLM-generated score (0-100)
4. **Top-N Selection**: Returns top 120 most relevant memories (default)
5. **Context Building**: Memories are formatted into prompt text

**Tunable Parameters** (in `settings.py`):
- `MEMORY_N_COUNT = 120` - Number of memories to retrieve
- `MEMORY_HP = [0, 1, 0.5]` - Weights for [recency, relevance, importance]
- `MEMORY_RECENCY_DECAY = 0.99` - Exponential decay factor
- `MEMORY_CURR_FILTER = "all"` - Filter by node type ("all", "observation", "reflection")

### LLM Interaction Pattern

The system uses **single-turn API calls with context injection**, NOT continuous conversation sessions:

- Each interaction = **one independent API call**
- Context is **manually constructed** in the prompt
- Previous interactions are **retrieved from agent's memory** (if relevant)
- No session state maintained with the LLM
- Multi-turn feel is achieved by including dialogue history in the prompt text

### Best Practices

1. **Call `agent.save()` regularly** to persist changes
2. **Use `remember()`** to store important interactions
3. **Use `reflect()`** periodically to generate insights
4. **Manage time_steps** sequentially for proper recency scoring

For detailed technical documentation, see the memory system implementation in `genagents/modules/memory_stream.py`.

---


## Agent Bank Access

Due to participant privacy concerns, the full agent bank containing over 1,000 generative agents based on real interviews is not publicly available at the moment. However, we plan to make aggregated responses on fixed tasks accessible for general research use in the coming months. Researchers interested in accessing individual responses on open tasks can request restricted access by contacting the authors and following a review process that ensures ethical considerations are met.

The codebase includes a demographic agent bank (`agent_bank/populations/gss_agents/`) with over 3,000 agents created using demographic information from the General Social Survey (GSS). *Note: The names and addresses are fictional.*


---


## References

Please refer to the original paper for detailed information on the methodology and findings:

- Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R., Liang, P., & Bernstein, M. S. (2024). *Generative Agent Simulations of 1,000 People*.

