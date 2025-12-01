# Complete Guide: Building and Using Generative Agents for Surveys and Behavioral Tests

This guide provides step-by-step instructions for creating generative agents and using them to answer text-based questions, including surveys and behavioral choice-making tests.

## Table of Contents

3. [Using Open Source Models](#using-open-source-models)
4. [Creating a New Agent](#creating-a-new-agent)
5. [Loading an Existing Agent](#loading-an-existing-agent)
6. [Interacting with Agents](#interacting-with-agents)
   - [Categorical Responses (Multiple Choice Surveys)](#categorical-responses-multiple-choice-surveys)
   - [Numerical Responses (Scale Questions)](#numerical-responses-scale-questions)
   - [Open-Ended Questions (Behavioral Choice-Making)](#open-ended-questions-behavioral-choice-making)
7. [Memory and Reflection](#memory-and-reflection)
8. [Saving and Loading Agents](#saving-and-loading-agents)
9. [Complete Examples](#complete-examples)
10. [Troubleshooting](#troubleshooting)

---

## Using Open Source Models

The genagents framework can be configured to use open-source models (like Qwen2.5-7B-Instruct) instead of OpenAI's GPT models. This section outlines the minimal, modular modifications needed.

### Overview

The system uses a provider pattern where model calls are routed through `gpt_structure.py`. To support open-source models, we need to:

1. **Modify settings** to specify the model provider
2. **Create a local model adapter** for open-source models
3. **Update the request function** to route to the appropriate provider

### Step 1: Install Additional Dependencies

For open-source models, you'll need additional packages:

```bash
pip install transformers torch sentence-transformers
```

**Note:** For GPU acceleration, install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
```

### Step 2: Update Settings Configuration

Modify `simulation_engine/settings.py` to include model provider settings:

```python
from pathlib import Path

# Model Provider Configuration
MODEL_PROVIDER = "openai"  # Options: "openai" or "local"
# MODEL_PROVIDER = "local"  # Uncomment to use local models

# OpenAI Configuration (only needed if MODEL_PROVIDER == "openai")
OPENAI_API_KEY = "sk-your-actual-api-key-here"
KEY_OWNER = "Your Name"

# Local Model Configuration (only needed if MODEL_PROVIDER == "local")
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Hugging Face model ID
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # For embeddings
DEVICE = "cuda"  # Options: "cuda", "cpu", or "mps" (for Apple Silicon)

# Optional: Hugging Face token (required for gated models)
HF_TOKEN = None  # Set to your HF token if needed: "hf_your_token_here"

# Debug mode
DEBUG = False
MAX_CHUNK_SIZE = 4

# LLM version (for OpenAI) or model identifier (for local)
LLM_VERS = "gpt-4o-mini"  # Used when MODEL_PROVIDER == "openai"
# LLM_VERS = "Qwen2.5-7B-Instruct"  # Used when MODEL_PROVIDER == "local"

BASE_DIR = f"{Path(__file__).resolve().parent.parent}"
POPULATIONS_DIR = f"{BASE_DIR}/agent_bank/populations"
LLM_PROMPT_DIR = f"{BASE_DIR}/simulation_engine/prompt_template"
```

### Step 3: Create Local Model Adapter

Create a new file `simulation_engine/local_model_adapter.py`:

```python
"""
Local Model Adapter for Open-Source LLMs
Supports models via Hugging Face Transformers
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import os

from simulation_engine.settings import *

# Global model cache to avoid reloading
_model_cache = {}
_tokenizer_cache = {}
_embedding_model_cache = None


def load_local_model(model_name: str = None, device: str = None):
    """
    Load a local model and tokenizer, with caching to avoid reloading.
    
    Args:
        model_name: Hugging Face model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
        device: Device to load model on ("cuda", "cpu", "mps")
    
    Returns:
        tuple: (model, tokenizer)
    """
    if model_name is None:
        model_name = LOCAL_MODEL_NAME
    if device is None:
        device = DEVICE
    
    # Check cache first
    cache_key = f"{model_name}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key], _tokenizer_cache[cache_key]
    
    print(f"Loading model: {model_name} on {device}")
    
    # Prepare authentication
    auth_kwargs = {}
    if HF_TOKEN:
        auth_kwargs['token'] = HF_TOKEN
        # For older transformers versions
        auth_kwargs['use_auth_token'] = HF_TOKEN
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **auth_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            **auth_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            **auth_kwargs
        )
        model = model.to(device)
    
    model.eval()
    
    # Cache the model
    _model_cache[cache_key] = model
    _tokenizer_cache[cache_key] = tokenizer
    
    print(f"Model loaded successfully!")
    return model, tokenizer


def load_embedding_model(model_name: str = None):
    """
    Load a sentence transformer model for embeddings.
    
    Args:
        model_name: Sentence transformer model name
    
    Returns:
        SentenceTransformer model
    """
    global _embedding_model_cache
    
    if model_name is None:
        model_name = LOCAL_EMBEDDING_MODEL
    
    if _embedding_model_cache is None:
        print(f"Loading embedding model: {model_name}")
        _embedding_model_cache = SentenceTransformer(model_name)
        print("Embedding model loaded!")
    
    return _embedding_model_cache


def local_model_request(prompt: str, 
                       model_name: str = None,
                       max_tokens: int = 1500,
                       temperature: float = 0.7) -> str:
    """
    Generate a response using a local model.
    
    Args:
        prompt: Input prompt text
        model_name: Model identifier (uses LOCAL_MODEL_NAME if None)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text response
    """
    try:
        model, tokenizer = load_local_model(model_name)
        device = DEVICE
        
        # Check if tokenizer supports chat templates (Qwen, Llama 3.1+, etc.)
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            # Use chat template for proper formatting
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            # Move inputs to device
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
        else:
            # Fallback: simple prompt formatting
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs if isinstance(inputs, dict) else {"input_ids": inputs},
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        if isinstance(inputs, dict):
            input_length = inputs["input_ids"].shape[1]
        else:
            input_length = inputs.shape[1]
        
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    except Exception as e:
        return f"GENERATION ERROR: {str(e)}"


def local_get_text_embedding(text: str, model_name: str = None) -> List[float]:
    """
    Generate text embedding using a local sentence transformer model.
    
    Args:
        text: Input text
        model_name: Embedding model name (uses LOCAL_EMBEDDING_MODEL if None)
    
    Returns:
        List of embedding values
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    
    text = text.replace("\n", " ").strip()
    model = load_embedding_model(model_name)
    embedding = model.encode(text, convert_to_numpy=True).tolist()
    
    return embedding
```

### Step 4: Modify gpt_structure.py

Update `simulation_engine/gpt_structure.py` to route requests based on provider:

```python
import openai
import time
import base64
from typing import List, Union

from simulation_engine.settings import *

# Only set OpenAI API key if using OpenAI provider
if MODEL_PROVIDER == "openai":
    openai.api_key = OPENAI_API_KEY

# Import local model adapter if using local models
if MODEL_PROVIDER == "local":
    from simulation_engine.local_model_adapter import (
        local_model_request,
        local_get_text_embedding
    )

# ... existing helper functions (print_run_prompts, generate_prompt) ...

def gpt_request(prompt: str, 
                model: str = "gpt-4o", 
                max_tokens: int = 1500) -> str:
    """Make a request to either OpenAI or local model based on MODEL_PROVIDER."""
    if MODEL_PROVIDER == "local":
        # Use local model
        return local_model_request(prompt, model_name=LOCAL_MODEL_NAME, 
                                  max_tokens=max_tokens, temperature=0.7)
    else:
        # Original OpenAI code
        if model == "o1-preview": 
            try:
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"GENERATION ERROR: {str(e)}"

        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"GENERATION ERROR: {str(e)}"


def get_text_embedding(text: str, 
                       model: str = "text-embedding-3-small") -> List[float]:
    """Generate embedding using either OpenAI or local model."""
    if MODEL_PROVIDER == "local":
        return local_get_text_embedding(text, model_name=LOCAL_EMBEDDING_MODEL)
    else:
        # Original OpenAI code
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        text = text.replace("\n", " ").strip()
        response = openai.embeddings.create(
            input=[text], model=model).data[0].embedding
        return response

# ... rest of the file remains the same ...
```

**Key Changes:**
- Add conditional import of local model adapter
- Modify `gpt_request()` to check `MODEL_PROVIDER` and route accordingly
- Modify `get_text_embedding()` to use local embeddings when `MODEL_PROVIDER == "local"`

### Step 5: Usage

Once configured, usage remains the same! The agent code doesn't need to change:

```python
from genagents.genagents import GenerativeAgent

# Create agent (works with both OpenAI and local models)
agent = GenerativeAgent()

# Set up agent
agent.update_scratch({
    "first_name": "John",
    "last_name": "Doe",
    "age": 30,
    "occupation": "Software Engineer"
})

# All interaction methods work the same way
questions = {
    "Do you enjoy coding?": ["Yes", "No", "Sometimes"]
}
response = agent.categorical_resp(questions)
print(response["responses"])
```

### Supported Models

**Tested Models:**
- ✅ `Qwen/Qwen2.5-7B-Instruct`
- ✅ `Qwen/Qwen2.5-14B-Instruct`
- ✅ `meta-llama/Llama-3.1-8B-Instruct`
- ✅ `meta-llama/Llama-3.1-70B-Instruct`

**Embedding Models:**
- ✅ `sentence-transformers/all-MiniLM-L6-v2` (default, fast, 384-dim)
- ✅ `sentence-transformers/all-mpnet-base-v2` (slower, better quality, 768-dim)


### Troubleshooting

**Issue: Model not found**
- Ensure model name matches Hugging Face identifier exactly
- Check if model requires authentication (set `HF_TOKEN`)

**Issue: Out of memory**
- Use smaller models (7B instead of 70B)
- Use CPU instead of CUDA: `DEVICE = "cpu"`
- Reduce `max_tokens` in requests

**Issue: Slow generation**
- Use GPU: `DEVICE = "cuda"`
- Use quantized models (4-bit/8-bit) for faster inference
- Consider using API for faster responses

**Issue: Embedding dimension mismatch**
- If switching embedding models, existing agent embeddings may be incompatible
- Re-create agents or use the same embedding model consistently

### Summary of Modifications

**Files to Create:**
1. `simulation_engine/local_model_adapter.py` - New file for local model handling

**Files to Modify:**
1. `simulation_engine/settings.py` - Add model provider configuration
2. `simulation_engine/gpt_structure.py` - Add routing logic (minimal changes)

**Dependencies to Add:**
- `transformers`
- `torch`
- `sentence-transformers`

**No Changes Required:**
- Agent creation code
- Interaction methods (`categorical_resp`, `numerical_resp`, `utterance`)
- Memory and reflection functions
- Any code in `genagents/` module

This modular approach ensures backward compatibility - existing code works with OpenAI, and switching to local models only requires configuration changes.

---

## Creating a New Agent

### Step 4: Initialize a New Agent

To create a new generative agent from scratch:

```python
from genagents.genagents import GenerativeAgent

# Create a new agent
agent = GenerativeAgent()

# The agent now has:
# - A unique ID (automatically generated)
# - An empty scratchpad (personal information)
# - An empty memory stream
```

### Step 5: Populate Agent's Personal Information

The "scratchpad" contains the agent's core identity and characteristics. Update it with personal information:

```python
# Update the agent's scratchpad with personal attributes
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

### Step 6: Add Initial Memories (Optional but Recommended)

Memories help the agent maintain consistency and context across interactions:

```python
# Add memories about the agent's experiences
agent.remember("Graduated from college with a degree in Computer Science", time_step=1)
agent.remember("Started working as a software engineer at a tech company", time_step=2)
agent.remember("Enjoys solving complex programming problems", time_step=3)
agent.remember("Prefers working independently but values team collaboration", time_step=4)
```

**Note:** `time_step` represents the chronological order of events. Use sequential integers starting from 1.

---

## Loading an Existing Agent

### Step 7: Load a Pre-existing Agent

If you have a saved agent or want to use the sample agent:

```python
from genagents.genagents import GenerativeAgent

# Load an existing agent from a directory
agent_folder = "agent_bank/populations/single_agent/01fd7d2a-0357-4c1b-9f3e-8eade2d537ae"
agent = GenerativeAgent(agent_folder=agent_folder)

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

---

## Interacting with Agents

The agent can answer three types of questions:
1. **Categorical** - Multiple choice questions
2. **Numerical** - Scale/rating questions
3. **Open-ended** - Free-form text responses

### Categorical Responses (Multiple Choice Surveys)

Use this for survey questions with predefined answer options.

#### Basic Example:

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

#### Single Question Example:

```python
# For a single question
single_question = {
    "Are you satisfied with your current job?": ["Very satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very dissatisfied"]
}

response = agent.categorical_resp(single_question)
print(f"Answer: {response['responses'][0]}")
print(f"Reasoning: {response['reasonings'][0]}")
```

#### Survey Example (Multiple Questions):

```python
# Comprehensive survey
survey_questions = {
    "How often do you exercise?": ["Daily", "Several times a week", "Weekly", "Rarely", "Never"],
    "What motivates you most at work?": ["Financial rewards", "Recognition", "Personal growth", "Helping others", "Creative freedom"],
    "How do you handle stress?": ["Exercise", "Meditation", "Talking to friends", "Work harder", "Avoid thinking about it"],
    "What is your biggest career goal?": ["Become a manager", "Start my own business", "Become an expert", "Work-life balance", "Financial security"]
}

survey_responses = agent.categorical_resp(survey_questions)

# Process responses
for i, (question, options) in enumerate(survey_questions.items()):
    answer = survey_responses["responses"][i]
    reasoning = survey_responses["reasonings"][i]
    print(f"\nQ: {question}")
    print(f"A: {answer}")
    print(f"Reasoning: {reasoning}")
```

### Numerical Responses (Scale Questions)

Use this for questions requiring numerical ratings on a scale.

#### Basic Example:

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
```

#### Float vs Integer Responses:

```python
# For integer responses (default)
questions_int = {
    "Rate your happiness level (1-10)": [1, 10]
}
response_int = agent.numerical_resp(questions_int, float_resp=False)
# Returns: {"responses": [7], ...}  # Integer

# For float responses (decimal values)
questions_float = {
    "Rate your confidence level (0.0-1.0)": [0.0, 1.0]
}
response_float = agent.numerical_resp(questions_float, float_resp=True)
# Returns: {"responses": [0.75], ...}  # Float
```

#### Likert Scale Example:

```python
# Likert scale questions (1-5 or 1-7)
likert_questions = {
    "I feel confident in my abilities": [1, 5],  # 1=Strongly Disagree, 5=Strongly Agree
    "I enjoy working in teams": [1, 5],
    "I prefer structured work environments": [1, 5],
    "I am satisfied with my career progress": [1, 5]
}

likert_responses = agent.numerical_resp(likert_questions, float_resp=False)

# Map responses to labels
scale_labels = {1: "Strongly Disagree", 2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly Agree"}

for i, (question, _) in enumerate(likert_questions.items()):
    score = likert_responses["responses"][i]
    label = scale_labels.get(score, "Unknown")
    print(f"{question}: {score} ({label})")
```

### Open-Ended Questions (Behavioral Choice-Making)

Use this for free-form responses, interviews, and behavioral scenarios.

#### Basic Example:

```python
# Create a dialogue format
# Each entry is a tuple: (speaker_name, message)
dialogue = [
    ("Interviewer", "Tell me about your favorite hobby."),
]

# Get agent's response
response = agent.utterance(dialogue)
print(f"Agent: {response}")
```

#### Multi-turn Conversation:

```python
# Build conversation incrementally
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

#### Behavioral Choice-Making Scenario:

```python
# Scenario: Decision-making test
scenario = [
    ("Researcher", "You are offered two job opportunities. Job A pays $80,000 but requires 60-hour weeks. Job B pays $60,000 but requires 40-hour weeks. Which would you choose and why?"),
]

response = agent.utterance(scenario)
print(f"Scenario: {scenario[0][1]}")
print(f"Agent's Response: {response}\n")

# Follow-up question
scenario.append((agent.get_fullname(), response))
scenario.append(("Researcher", "What factors are most important to you in making this decision?"))
follow_up = agent.utterance(scenario)
print(f"Follow-up: What factors are most important to you in making this decision?")
print(f"Agent's Response: {follow_up}")
```

#### Complex Behavioral Test:

```python
# Multi-step behavioral assessment
behavioral_test = []

# Step 1: Ethical dilemma
behavioral_test.append(("Researcher", 
    "You discover a colleague has been taking credit for your work. How would you handle this situation?"))

response1 = agent.utterance(behavioral_test)
behavioral_test.append((agent.get_fullname(), response1))
print(f"Q1: {behavioral_test[0][1]}")
print(f"A1: {response1}\n")

# Step 2: Risk assessment
behavioral_test.append(("Researcher",
    "You have the opportunity to invest in a startup. There's a 70% chance you'll lose your investment, but a 30% chance of 10x returns. What would you do?"))

response2 = agent.utterance(behavioral_test)
behavioral_test.append((agent.get_fullname(), response2))
print(f"Q2: {behavioral_test[2][1]}")
print(f"A2: {response2}\n")

# Step 3: Social situation
behavioral_test.append(("Researcher",
    "At a party, you see someone being excluded from a group conversation. How do you respond?"))

response3 = agent.utterance(behavioral_test)
print(f"Q3: {behavioral_test[4][1]}")
print(f"A3: {response3}")
```

#### With Additional Context:

```python
# Provide context for the conversation
dialogue = [
    ("Interviewer", "Tell me about a challenging situation you faced at work."),
]

# Add context (optional third parameter)
context = "This is a job interview for a software engineering position."
response = agent.utterance(dialogue, context=context)
print(response)
```

---

## Memory and Reflection

### Adding Memories

Memories help agents maintain consistency and learn from interactions:

```python
# Add a memory about an interaction
agent.remember("Participated in a survey about work preferences", time_step=5)
agent.remember("Expressed preference for remote work", time_step=6)
agent.remember("Indicated high satisfaction with current role", time_step=7)
```

### Reflection

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

---

## Saving and Loading Agents

### Saving an Agent

Save the agent's complete state for later use:

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

### Loading a Saved Agent

```python
# Load the saved agent
agent = GenerativeAgent(agent_folder="agent_bank/populations/my_agent")

# Continue interacting
response = agent.categorical_resp({"Favorite color?": ["Red", "Blue", "Green"]})
```

---

## Complete Examples

### Example 1: Complete Survey Workflow

```python
from genagents.genagents import GenerativeAgent

# Step 1: Create agent
agent = GenerativeAgent()

# Step 2: Set up personal information
agent.update_scratch({
    "first_name": "Sarah",
    "last_name": "Johnson",
    "age": 28,
    "occupation": "Marketing Manager",
    "education": "Master's degree",
    "personality": "Extroverted, creative, goal-oriented"
})

# Step 3: Add some memories
agent.remember("Started career in marketing 5 years ago", time_step=1)
agent.remember("Recently promoted to manager role", time_step=2)
agent.remember("Enjoys creative problem-solving", time_step=3)

# Step 4: Conduct a survey
survey = {
    "How satisfied are you with your current role?": 
        ["Very satisfied", "Satisfied", "Neutral", "Dissatisfied"],
    "What is your primary motivation at work?": 
        ["Career growth", "Financial rewards", "Work-life balance", "Creative fulfillment"],
    "Rate your work stress level (1-10)": [1, 10],
    "How do you prefer to receive feedback?": 
        ["Formal reviews", "Regular check-ins", "Informal conversations", "Written feedback"]
}

# Get categorical responses
categorical_questions = {k: v for k, v in survey.items() if isinstance(v, list) and not isinstance(v[0], int)}
categorical_responses = agent.categorical_resp(categorical_questions)

# Get numerical responses
numerical_questions = {k: v for k, v in survey.items() if isinstance(v, list) and isinstance(v[0], int)}
if numerical_questions:
    numerical_responses = agent.numerical_resp(numerical_questions, float_resp=False)

# Step 5: Display results
print("=== Survey Results ===")
for i, (q, _) in enumerate(categorical_questions.items()):
    print(f"\nQ: {q}")
    print(f"A: {categorical_responses['responses'][i]}")

if numerical_questions:
    for i, (q, _) in enumerate(numerical_questions.items()):
        print(f"\nQ: {q}")
        print(f"A: {numerical_responses['responses'][i]}")

# Step 6: Save agent
agent.save("agent_bank/populations/sarah_johnson")
```

### Example 2: Behavioral Choice-Making Test

```python
from genagents.genagents import GenerativeAgent

# Create and configure agent
agent = GenerativeAgent()
agent.update_scratch({
    "first_name": "Michael",
    "last_name": "Chen",
    "age": 35,
    "occupation": "Financial Analyst",
    "personality": "Analytical, risk-averse, detail-oriented",
    "values": "Financial security, family, stability"
})

# Behavioral scenario 1: Risk tolerance
scenario1 = [
    ("Researcher", 
     "You have $10,000 to invest. Option A: Guaranteed 5% return. Option B: 50% chance of 20% return, 50% chance of losing 10%. Which do you choose?")
]
response1 = agent.utterance(scenario1)
scenario1.append((agent.get_fullname(), response1))
print(f"Scenario 1 Response: {response1}\n")

# Behavioral scenario 2: Work-life balance
scenario2 = scenario1.copy()
scenario2.append(("Researcher",
    "Your boss offers you a promotion that requires 20% more hours and frequent travel, but comes with a 30% salary increase. How do you respond?"))
response2 = agent.utterance(scenario2)
print(f"Scenario 2 Response: {response2}\n")

# Save memories of these interactions
agent.remember("Participated in risk tolerance assessment", time_step=1)
agent.remember("Participated in work-life balance scenario", time_step=2)

# Reflect on decision-making patterns
agent.reflect(anchor="decision making and risk", time_step=3)

# Save agent
agent.save("agent_bank/populations/michael_chen")
```

### Example 3: Multi-Method Assessment

```python
from genagents.genagents import GenerativeAgent

# Create agent
agent = GenerativeAgent()
agent.update_scratch({
    "first_name": "Emma",
    "last_name": "Williams",
    "age": 42,
    "occupation": "Teacher",
    "education": "Master's in Education",
    "personality": "Patient, empathetic, organized"
})

# Part 1: Categorical survey
categorical_survey = {
    "How do you handle classroom conflicts?": 
        ["Address immediately", "Wait and observe", "Refer to administration", "Discuss privately"],
    "What teaching method do you prefer?": 
        ["Lecture-based", "Interactive discussion", "Hands-on activities", "Mixed approach"]
}
cat_responses = agent.categorical_resp(categorical_survey)

# Part 2: Numerical ratings
numerical_survey = {
    "Rate your confidence in managing difficult students (1-10)": [1, 10],
    "How effective is your current teaching approach? (1-5)": [1, 5]
}
num_responses = agent.numerical_resp(numerical_survey, float_resp=False)

# Part 3: Open-ended behavioral question
behavioral_q = [
    ("Researcher",
     "Describe a time when you had to adapt your teaching style to help a struggling student. What did you do and what was the outcome?")
]
behavioral_response = agent.utterance(behavioral_q)

# Part 4: Follow-up interview
interview = behavioral_q.copy()
interview.append((agent.get_fullname(), behavioral_response))
interview.append(("Researcher", "How do you balance individual student needs with curriculum requirements?"))
follow_up = agent.utterance(interview)

# Display all results
print("=== Categorical Responses ===")
for i, (q, _) in enumerate(categorical_survey.items()):
    print(f"{q}: {cat_responses['responses'][i]}")

print("\n=== Numerical Responses ===")
for i, (q, _) in enumerate(numerical_survey.items()):
    print(f"{q}: {num_responses['responses'][i]}")

print("\n=== Behavioral Response ===")
print(behavioral_response)

print("\n=== Follow-up Response ===")
print(follow_up)

# Save agent
agent.save("agent_bank/populations/emma_williams")
```

---

## Additional Resources

- **Main README:** See `README.md` for project overview
- **Sample Agent:** Explore `agent_bank/populations/single_agent/` for an example
- **Prompt Templates:** Check `simulation_engine/prompt_template/` to understand how prompts are structured


