# Agent Memory Mechanics Documentation

This document details the internal mechanics of how generative agents manage and utilize their memory system.

## Table of Contents

1. [Memory Node Creation Events](#1-memory-node-creation-events)
2. [Memory Storage Before agent.save()](#2-memory-storage-before-agentsave)
3. [Memory Node Types](#3-memory-node-types)
4. [LLM Interaction Pattern](#4-llm-interaction-pattern)
5. [Memory Usage During Interactions](#5-memory-usage-during-interactions)

---

## 1. Memory Node Creation Events

### Overview

New memory nodes are added to the agent's memory stream through **two primary methods**:

1. **`agent.remember()`** - Creates observation nodes
2. **`agent.reflect()`** - Creates reflection nodes

### 1.1 The `remember()` Method

**Location:** `genagents/genagents.py` (line 99-109) → `genagents/modules/memory_stream.py` (line 459-461)

**Process:**
```python
def remember(self, content, time_step=0):
    # 1. Generate importance score using LLM
    score = generate_importance_score([content])[0]
    
    # 2. Add node with type "observation"
    self._add_node(time_step, "observation", content, score, None)
```

**What happens:**
- Takes a content string (the memory to store)
- Calls LLM to generate an importance score (0-100)
- Creates a new node with `node_type="observation"`
- Stores the embedding for semantic search
- Adds to `self.memory_stream.seq_nodes` list

**Example:**
```python
agent.remember("I graduated from college with honors", time_step=1)
# Creates: observation node with importance score from LLM
```

### 1.2 The `reflect()` Method

**Location:** `genagents/genagents.py` (line 112-121) → `genagents/modules/memory_stream.py` (line 464-473)

**Process:**
```python
def reflect(self, anchor, reflection_count=5, retrieval_count=120, time_step=0):
    # 1. Retrieve relevant memories based on anchor
    records = self.retrieve([anchor], time_step, retrieval_count)[anchor]
    record_ids = [i.node_id for i in records]
    
    # 2. Generate reflections using LLM
    reflections = generate_reflection(records, anchor, reflection_count)
    
    # 3. Generate importance scores for each reflection
    scores = generate_importance_score(reflections)
    
    # 4. Create reflection nodes (one per reflection)
    for count, reflection in enumerate(reflections): 
        self._add_node(time_step, "reflection", reflections[count], 
                      scores[count], record_ids)
```

**What happens:**
- Takes an anchor topic/keyword
- Retrieves relevant existing memories (up to `retrieval_count`, default 120)
- Sends retrieved memories + anchor to LLM to generate insights
- LLM generates multiple reflections (default: 5)
- Each reflection gets an importance score from LLM
- Creates multiple nodes with `node_type="reflection"`
- Each reflection node has `pointer_id` linking to source observation nodes

**Example:**
```python
agent.reflect(anchor="career decisions", time_step=10)
# Creates: 5 reflection nodes (default) with insights about career decisions
```

### 1.3 Internal `_add_node()` Method

**Location:** `genagents/modules/memory_stream.py` (line 430-456)

This is the **only** method that actually creates memory nodes. Both `remember()` and `reflect()` call this internally.

```python
def _add_node(self, time_step, node_type, content, importance, pointer_id):
    # Create node dictionary
    node_dict = {
        "node_id": len(self.seq_nodes),  # Auto-incrementing ID
        "node_type": node_type,           # "observation" or "reflection"
        "content": content,               # The memory text
        "importance": importance,         # LLM-generated score (0-100)
        "created": time_step,             # When created
        "last_retrieved": time_step,      # When last accessed
        "pointer_id": pointer_id          # Links to parent nodes (for reflections)
    }
    
    # Create ConceptNode object
    new_node = ConceptNode(node_dict)
    
    # Add to memory stream
    self.seq_nodes += [new_node]
    self.id_to_node[new_node.node_id] = new_node
    
    # Generate and store embedding
    self.embeddings[content] = get_text_embedding(content)
```

### Summary

**Answer to Question 1:** Yes, `remember()` and `reflect()` are the **only two events** that trigger new memory nodes. Both call the internal `_add_node()` method, which is the single point of memory creation.

**Key Points:**
- No automatic memory creation during interactions (categorical_resp, numerical_resp, utterance)
- Memories must be explicitly added via `remember()` or `reflect()`
- Each node creation triggers an LLM call for importance scoring
- Each node creation triggers an embedding generation for semantic search

---

## 2. Memory Storage Before agent.save()

### Overview

Before `agent.save()` is called, all memory data is stored **in-memory** as Python objects within the `GenerativeAgent` instance.

### 2.1 Memory Storage Structure

**Location:** `genagents/genagents.py` (line 11-34)

When an agent is created:
```python
class GenerativeAgent:
    def __init__(self, agent_folder=None):
        if agent_folder:
            # Load from disk
            self.memory_stream = MemoryStream(nodes, embeddings)
        else:
            # Create new agent
            self.memory_stream = MemoryStream([], {})  # Empty memory stream
```

### 2.2 In-Memory Storage Components

**Location:** `genagents/modules/memory_stream.py` (line 316-326)

The `MemoryStream` class stores data in three main structures:

```python
class MemoryStream:
    def __init__(self, nodes, embeddings):
        # 1. Sequential list of all memory nodes
        self.seq_nodes = []  # List of ConceptNode objects
        
        # 2. Dictionary mapping node_id to node object (for fast lookup)
        self.id_to_node = {}  # {node_id: ConceptNode}
        
        # 3. Dictionary mapping content to embedding vector
        self.embeddings = {}  # {content_string: [embedding_vector]}
```

### 2.3 Data Flow

**During Agent Operations:**

1. **Node Creation:**
   ```python
   agent.remember("Some memory", time_step=1)
   # → Adds to self.memory_stream.seq_nodes
   # → Adds to self.memory_stream.id_to_node
   # → Adds to self.memory_stream.embeddings
   ```

2. **In-Memory State:**
   - `self.memory_stream.seq_nodes`: Python list in RAM
   - `self.memory_stream.id_to_node`: Python dict in RAM
   - `self.memory_stream.embeddings`: Python dict in RAM

3. **Persistence:**
   ```python
   agent.save("path/to/directory")
   # → Writes seq_nodes to nodes.json
   # → Writes embeddings to embeddings.json
   # → Writes scratch to scratch.json
   ```

### 2.4 What Happens on agent.save()

**Location:** `genagents/genagents.py` (line 53-87)

```python
def save(self, save_directory):
    # Create directory structure
    create_folder_if_not_there(f"{storage}/memory_stream")
    
    # Serialize and save memory nodes
    with open(f"{storage}/memory_stream/nodes.json", "w") as json_file:
        json.dump([node.package() for node in self.memory_stream.seq_nodes], 
                 json_file, indent=2)
    
    # Serialize and save embeddings
    with open(f"{storage}/memory_stream/embeddings.json", "w") as json_file:
        json.dump(self.memory_stream.embeddings, json_file)
    
    # Save scratchpad
    with open(f"{storage}/scratch.json", "w") as json_file:
        json.dump(self.scratch, json_file, indent=2)
    
    # Save metadata
    with open(f"{storage}/meta.json", "w") as json_file:
        json.dump(self.package(), json_file, indent=2)
```

### Summary

**Answer to Question 2:** Before `agent.save()` is called, all memory data is stored in **Python variables** within the agent object:

- **`self.memory_stream.seq_nodes`** - List of ConceptNode objects (in RAM)
- **`self.memory_stream.id_to_node`** - Dictionary for node lookup (in RAM)
- **`self.memory_stream.embeddings`** - Dictionary of content→embedding mappings (in RAM)

**Important Notes:**
- Data is **volatile** - lost if program crashes before `save()`
- Data is **not persisted** until `agent.save()` is explicitly called
- All operations (remember, reflect, retrieve) work on in-memory data
- `agent.save()` serializes everything to JSON files on disk

---

## 3. Memory Node Types

### Overview

Each memory node has a `node_type` field that categorizes the memory. There are **exactly 2 types** in the system.

### 3.1 Node Types

1. **`"observation"`** - Direct memories/experiences
2. **`"reflection"`** - LLM-generated insights derived from observations

### 3.2 Type Assignment

**Location:** `genagents/modules/memory_stream.py` (line 430-473)

Node type is assigned **at creation time** based on which method is called:

#### Type: `"observation"`

**Assigned when:** `agent.remember()` is called

```python
def remember(self, content, time_step=0):
    score = generate_importance_score([content])[0]
    self._add_node(time_step, "observation", content, score, None)
    #                                    ^^^^^^^^^^^^
    #                                    Hard-coded as "observation"
```

**Characteristics:**
- Created directly from user-provided content
- `pointer_id` is always `None` (no parent nodes)
- Represents raw experiences, events, or facts
- Stored exactly as provided (no LLM modification of content)

**Example:**
```python
agent.remember("I graduated from MIT in 2020", time_step=1)
# Creates: node_type="observation"
```

#### Type: `"reflection"`

**Assigned when:** `agent.reflect()` is called

```python
def reflect(self, anchor, reflection_count=5, retrieval_count=120, time_step=0):
    records = self.retrieve([anchor], time_step, retrieval_count)[anchor]
    record_ids = [i.node_id for i in records]
    reflections = generate_reflection(records, anchor, reflection_count)
    scores = generate_importance_score(reflections)
    
    for count, reflection in enumerate(reflections): 
        self._add_node(time_step, "reflection", reflections[count], 
                      scores[count], record_ids)
        #                ^^^^^^^^^^
        #                Hard-coded as "reflection"
```

**Characteristics:**
- Created from LLM-generated insights
- Content is **generated by LLM**, not user-provided
- `pointer_id` contains list of source observation node IDs
- Represents synthesized insights, patterns, or conclusions
- Multiple reflections can be created from one `reflect()` call

**Example:**
```python
agent.reflect(anchor="career path", time_step=10)
# Creates: 5 nodes with node_type="reflection" (default reflection_count=5)
# Each reflection node has pointer_id linking to source observations
```

### 3.3 Node Type Usage

**Filtering by Type:**

**Location:** `genagents/modules/memory_stream.py` (line 346-377)

The `retrieve()` method can filter nodes by type:

```python
def retrieve(self, focal_points, time_step, n_count=120, curr_filter="all", ...):
    # curr_filter options:
    # - "all": Include both observations and reflections
    # - "observation": Only observation nodes
    # - "reflection": Only reflection nodes
    
    if curr_filter == "all": 
        curr_nodes = self.seq_nodes
    else: 
        for curr_node in self.seq_nodes: 
            if curr_node.node_type == curr_filter: 
                curr_nodes += [curr_node]
```

**Counting Observations:**

**Location:** `genagents/modules/memory_stream.py` (line 329-343)

```python
def count_observations(self):
    """Count observations (excludes reflections)"""
    count = 0
    for i in self.seq_nodes: 
        if i.node_type == "observation": 
            count += 1
    return count
```

### 3.4 Node Structure

Each node contains:

```python
{
    "node_id": 0,                    # Unique identifier
    "node_type": "observation",       # "observation" or "reflection"
    "content": "...",                 # The memory text
    "importance": 85,                 # LLM-generated score (0-100)
    "created": 1,                     # Time step when created
    "last_retrieved": 1,             # Time step when last accessed
    "pointer_id": null                # For observations: null
                                      # For reflections: list of parent node IDs
}
```

### Summary

**Answer to Question 3:** There are **exactly 2 node types**:

1. **`"observation"`** - Assigned when `agent.remember()` is called
   - User-provided content
   - `pointer_id = None`
   
2. **`"reflection"`** - Assigned when `agent.reflect()` is called
   - LLM-generated content
   - `pointer_id = [list of source node IDs]`

**Type Assignment:** Hard-coded in the `_add_node()` call - no dynamic assignment logic.

---

## 4. LLM Interaction Pattern

### Overview

The genagents framework uses a **single-turn API call pattern** with **context injection** to simulate multi-turn conversations. There is **no continuous conversation session** maintained with the LLM.

### 4.1 Single-Turn Pattern

**Location:** `simulation_engine/gpt_structure.py` (line 53-78)

Every LLM interaction is a **fresh, independent API call**:

```python
def gpt_request(prompt: str, model: str = "gpt-4o", max_tokens: int = 1500) -> str:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],  # Single message
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content
```

**Key Characteristics:**
- Each call sends a **single message** with `role="user"`
- No conversation history maintained by the API
- Each call is **stateless** - no session continuity
- Response is returned immediately, no streaming or session

### 4.2 Context Injection for Multi-Turn Simulation

While the API calls are single-turn, the system **simulates multi-turn conversations** by:

1. **Building context in the prompt**
2. **Including retrieved memories**
3. **Formatting dialogue history as text**

#### Example: Categorical Response

**Location:** `genagents/modules/interaction.py` (line 84-88)

```python
def categorical_resp(agent, questions):
    anchor = " ".join(list(questions.keys()))
    agent_desc = _main_agent_desc(agent, anchor)  # Retrieves relevant memories
    return run_gpt_generate_categorical_resp(
        agent_desc, questions, "1", LLM_VERS)[0]
```

**What `_main_agent_desc()` does:**
```python
def _main_agent_desc(agent, anchor): 
    agent_desc = ""
    agent_desc += f"Self description: {agent.get_self_description()}\n==\n"
    agent_desc += f"Other observations about the subject:\n\n"
    
    # Retrieve relevant memories (up to 120)
    retrieved = agent.memory_stream.retrieve([anchor], 0, n_count=120)
    if len(retrieved) == 0:
        return agent_desc
    
    nodes = list(retrieved.values())[0]
    for node in nodes:
        agent_desc += f"{node.content}\n"  # Append memory content
    
    return agent_desc
```

**Result:** A single prompt containing:
- Agent's scratchpad (personal info)
- Relevant memories (retrieved via semantic search)
- The current question

#### Example: Open-Ended Utterance

**Location:** `genagents/modules/interaction.py` (line 177-186)

```python
def utterance(agent, curr_dialogue, context): 
    # Format dialogue history as text
    str_dialogue = ""
    for row in curr_dialogue:
        str_dialogue += f"[{row[0]}]: {row[1]}\n"
    str_dialogue += f"[{agent.get_fullname()}]: [Fill in]\n"
    
    # Retrieve relevant memories
    anchor = str_dialogue
    agent_desc = _utterance_agent_desc(agent, anchor)
    
    # Single API call with full context
    return run_gpt_generate_utterance(
        agent_desc, str_dialogue, context, "1", LLM_VERS)[0]
```

**What happens:**
1. Dialogue history is formatted as text string
2. Relevant memories are retrieved based on dialogue content
3. Everything is combined into a single prompt
4. One API call generates the response

### 4.3 No Session State

**Important:** The system does **NOT**:
- Maintain a conversation session with the LLM
- Store conversation history in the LLM's context window
- Use streaming or chat completion sessions
- Keep any state between API calls

**Instead:**
- Each interaction is **completely independent**
- Context is **manually constructed** in the prompt
- Previous interactions are **retrieved from memory** if relevant
- Memory retrieval uses **semantic search** (embeddings) to find relevant past interactions

### 4.4 Interaction Flow Diagram

```
User calls: agent.categorical_resp(questions)
    ↓
1. Build anchor from questions
    ↓
2. Retrieve relevant memories (semantic search)
    ↓
3. Build prompt with:
   - Agent scratchpad
   - Retrieved memories
   - Current questions
    ↓
4. Single API call to LLM
    ↓
5. Return response
    ↓
[No state maintained - next call starts fresh]
```

### 4.5 Comparison: Single-Turn vs Multi-Turn

| Aspect | Genagents Approach | True Multi-Turn Session |
|--------|-------------------|------------------------|
| API Calls | One per interaction | One session, multiple messages |
| Context | Manually built in prompt | Maintained by API |
| History | Retrieved from memory | In API's context window |
| State | Stateless | Stateful session |
| Memory | Agent's memory stream | API's conversation history |

### Summary

**Answer to Question 4:** The system uses **single-turn API calls with context injection**, NOT continuous conversation sessions.

**Key Points:**
- Each interaction = **one independent API call**
- Context is **manually constructed** in the prompt
- Previous interactions are **retrieved from agent's memory** (if relevant)
- No session state maintained with the LLM
- Multi-turn feel is achieved by including dialogue history in the prompt text

**Example:**
```python
# This looks like multi-turn, but each call is independent:
dialogue = [("Interviewer", "Hello")]
response1 = agent.utterance(dialogue)  # API call #1

dialogue.append((agent.get_fullname(), response1))
dialogue.append(("Interviewer", "Tell me more"))
response2 = agent.utterance(dialogue)  # API call #2 (fresh call, but includes history in prompt)
```

---

## 5. Memory Usage During Interactions

### Overview

**Yes, memory is automatically utilized** in every agent interaction (categorical_resp, numerical_resp, utterance). The system automatically retrieves relevant memories and appends them to the agent's description in the prompt sent to the LLM.

### 5.1 Automatic Memory Retrieval

**Location:** `genagents/modules/interaction.py` (lines 17-43)

Every interaction method automatically calls memory retrieval:

```python
def categorical_resp(agent, questions):
    anchor = " ".join(list(questions.keys()))
    agent_desc = _main_agent_desc(agent, anchor)  # ← Memory retrieval happens here
    return run_gpt_generate_categorical_resp(agent_desc, questions, ...)

def numerical_resp(agent, questions, float_resp):
    anchor = " ".join(list(questions.keys()))
    agent_desc = _main_agent_desc(agent, anchor)  # ← Memory retrieval happens here
    return run_gpt_generate_numerical_resp(agent_desc, questions, ...)

def utterance(agent, curr_dialogue, context):
    anchor = str_dialogue
    agent_desc = _utterance_agent_desc(agent, anchor)  # ← Memory retrieval happens here
    return run_gpt_generate_utterance(agent_desc, str_dialogue, context, ...)
```

**Key Point:** Memory retrieval is **automatic and transparent** - you don't need to manually call it.

### 5.2 Agent Description Format

**Location:** `genagents/modules/interaction.py` (lines 17-28, 31-43)

The agent description is built in a specific format:

```python
def _main_agent_desc(agent, anchor): 
    agent_desc = ""
    
    # Part 1: Scratchpad (personal information)
    agent_desc += f"Self description: {agent.get_self_description()}\n==\n"
    
    # Part 2: Retrieved memories
    agent_desc += f"Other observations about the subject:\n\n"
    
    # Retrieve relevant memories (up to 120 by default)
    retrieved = agent.memory_stream.retrieve([anchor], 0, n_count=120)
    if len(retrieved) == 0:
        return agent_desc
    
    # Append each memory's content
    nodes = list(retrieved.values())[0]
    for node in nodes:
        agent_desc += f"{node.content}\n"
    
    return agent_desc
```

**Exact Format in Prompt:**

```
Self description: {'first_name': 'John', 'last_name': 'Doe', 'age': 30, ...}
==
Other observations about the subject:

[Memory 1 content]
[Memory 2 content]
[Memory 3 content]
...
[Memory N content]
```

**Note:** The scratchpad is converted to string via `agent.get_self_description()`, which calls `str(self.scratch)`, resulting in a Python dictionary string representation.

### 5.3 Memory Selection Process

**Location:** `genagents/modules/memory_stream.py` (line 346-427)

Memories are selected through a **three-dimensional scoring system**:

#### Step 1: Calculate Individual Scores

For each memory node, three scores are calculated:

**1. Recency Score:**
```python
# Formula: recency_decay ^ (max_timestep - node.last_retrieved)
recency_decay = 0.99  # Fixed decay factor
recency_score = 0.99 ** (max_timestep - node.last_retrieved)
```
- Range: 0 to 1 (1 = most recent)
- Based on `last_retrieved` time step
- Exponential decay: older memories get lower scores

**2. Relevance Score:**
```python
# Formula: cosine_similarity(query_embedding, memory_embedding)
relevance_score = cos_sim(query_embedding, memory_embedding)
```
- Range: -1 to 1 (normalized to 0-1)
- Based on semantic similarity between query and memory
- Uses embeddings for semantic matching

**3. Importance Score:**
```python
# Directly from node.importance (LLM-generated at creation time)
importance_score = node.importance / 100.0  # Normalized from 0-100 to 0-1
```
- Range: 0 to 1
- Pre-computed when memory was created
- Based on LLM's assessment of memory importance

#### Step 2: Normalize Scores

All three scores are normalized to [0, 1] range:

```python
recency_out = normalize_dict_floats(recency_scores, 0, 1)
importance_out = normalize_dict_floats(importance_scores, 0, 1)
relevance_out = normalize_dict_floats(relevance_scores, 0, 1)
```

#### Step 3: Weighted Combination

Final score combines all three dimensions:

```python
final_score = (recency_weight × recency_score) + 
              (relevance_weight × relevance_score) + 
              (importance_weight × importance_score)
```

#### Step 4: Top-N Selection

Memories are ranked by final score and top N are selected:

```python
# Select top n_count memories (default: 120)
top_memories = top_highest_x_values(final_scores, n_count=120)
```

#### Step 5: Chronological Sorting

Selected memories are sorted by creation time (oldest first):

```python
master_nodes = sorted(master_nodes, 
                     key=lambda node: node.created, reverse=False)
```

### 5.4 Tunable Variables

**Location:** `simulation_engine/settings.py` (global defaults) and `genagents/modules/memory_stream.py` (function parameters)

All tunable parameters are now centralized in `simulation_engine/settings.py` for easy adjustment. The `retrieve()` method accepts these parameters with global defaults:

#### 5.4.1 Hyperparameters (hp)

**Global Default:** `MEMORY_HP = [0, 1, 0.5]` in `settings.py`

**Parameter:** `hp=[recency_w, relevance_w, importance_w]`

**Description:** Weights for the three scoring dimensions

**Usage:**
```python
# Use global default from settings.py
retrieved = agent.memory_stream.retrieve([anchor])

# Override with custom weights
retrieved = agent.memory_stream.retrieve([anchor], hp=[0.5, 0.3, 0.2])

# Equal weighting
retrieved = agent.memory_stream.retrieve([anchor], hp=[0.33, 0.33, 0.34])
```

**To Change Global Default:** Edit `MEMORY_HP` in `simulation_engine/settings.py`

#### 5.4.2 Number of Memories (n_count)

**Global Default:** `MEMORY_N_COUNT = 120` in `settings.py`

**Parameter:** `n_count=120`

**Description:** Maximum number of memories to retrieve

**Usage:**
```python
# Use global default from settings.py
retrieved = agent.memory_stream.retrieve([anchor])

# Override with custom count
retrieved = agent.memory_stream.retrieve([anchor], n_count=50)
```

**To Change Global Default:** Edit `MEMORY_N_COUNT` in `simulation_engine/settings.py`

#### 5.4.3 Node Type Filter (curr_filter)

**Global Default:** `MEMORY_CURR_FILTER = "all"` in `settings.py`

**Parameter:** `curr_filter="all"`

**Options:** `"all"`, `"observation"`, `"reflection"`

**Description:** Filter memories by node type

**Usage:**
```python
# Use global default from settings.py
retrieved = agent.memory_stream.retrieve([anchor])

# Override with custom filter
retrieved = agent.memory_stream.retrieve([anchor], curr_filter="observation")
```

**To Change Global Default:** Edit `MEMORY_CURR_FILTER` in `simulation_engine/settings.py`

#### 5.4.4 Recency Decay Factor

**Global Default:** `MEMORY_RECENCY_DECAY = 0.99` in `settings.py`

**Description:** Exponential decay factor for recency scoring

**Formula Impact:**
- `0.99` means ~1% decay per time step
- `0.95` would mean ~5% decay per time step (faster decay)
- `0.999` would mean ~0.1% decay per time step (slower decay)

**To Change Global Default:** Edit `MEMORY_RECENCY_DECAY` in `simulation_engine/settings.py`

#### 5.4.5 Time Step

**Global Default:** `MEMORY_TIME_STEP = 0` in `settings.py`

**Parameter:** `time_step=0`

**Description:** Current time step for recency calculation

**Usage:**
```python
# Use global default from settings.py
retrieved = agent.memory_stream.retrieve([anchor])

# Override with custom time step
retrieved = agent.memory_stream.retrieve([anchor], time_step=10)
```

**To Change Global Default:** Edit `MEMORY_TIME_STEP` in `simulation_engine/settings.py`

#### 5.4.6 Reflection Count

**Global Default:** `MEMORY_REFLECTION_COUNT = 5` in `settings.py`

**Parameter:** `reflection_count=5` in `agent.reflect()`

**Description:** Number of reflection insights to generate

**Usage:**
```python
# Use global default
agent.reflect(anchor="career decisions")

# Override with custom count
agent.reflect(anchor="career decisions", reflection_count=10)
```

**To Change Global Default:** Edit `MEMORY_REFLECTION_COUNT` in `simulation_engine/settings.py`

#### 5.4.7 Reflection Retrieval Count

**Global Default:** `MEMORY_RETRIEVAL_COUNT = 120` in `settings.py`

**Parameter:** `retrieval_count=120` in `agent.reflect()`

**Description:** Number of memories to retrieve when generating reflections

**Usage:**
```python
# Use global default
agent.reflect(anchor="career decisions")

# Override with custom count
agent.reflect(anchor="career decisions", retrieval_count=50)
```

**To Change Global Default:** Edit `MEMORY_RETRIEVAL_COUNT` in `simulation_engine/settings.py`

### 5.5 Memory Retrieval Flow Diagram

```
User calls: agent.categorical_resp(questions)
    ↓
1. Build anchor from questions
    ↓
2. Call _main_agent_desc(agent, anchor)
    ↓
3. Retrieve memories:
   a. Calculate recency scores (exponential decay)
   b. Calculate relevance scores (cosine similarity)
   c. Calculate importance scores (from node.importance)
   d. Normalize all scores to [0, 1]
   e. Weighted combination: hp[0]×recency + hp[1]×relevance + hp[2]×importance
   f. Select top 120 memories
   g. Sort chronologically
    ↓
4. Build agent description:
   - Scratchpad (self description)
   - Retrieved memories (up to 120)
    ↓
5. Format into prompt template
    ↓
6. Send to LLM
    ↓
7. Return response
```

### 5.6 Example: Complete Prompt Structure

For a categorical response, the final prompt looks like:

```
[Prompt Template Header]

Self description: {'first_name': 'Sarah', 'last_name': 'Johnson', 'age': 28, 'occupation': 'Marketing Manager', 'education': 'Master\'s degree', 'personality': 'Extroverted, creative, goal-oriented'}
==
Other observations about the subject:

Started career in marketing 5 years ago
Recently promoted to manager role
Enjoys creative problem-solving
Participated in a survey about work preferences
Expressed preference for remote work
Indicated high satisfaction with current role
[Additional memories up to 120 total...]

=====

Task: What you see above is an interview transcript. Based on the interview transcript, I want you to predict the participant's survey responses...

Here is the question: 

Q: How satisfied are you with your current role?
Option: ['Very satisfied', 'Satisfied', 'Neutral', 'Dissatisfied']
```

### 5.7 Customization Options

**All Parameters Are Now Tunable:**

All memory parameters are now centralized in `simulation_engine/settings.py` and can be easily adjusted:

1. **Global Defaults:** Edit values in `settings.py` to change defaults for all interactions
2. **Per-Call Overrides:** All functions accept parameters to override defaults when needed

**To Customize Globally:**

Edit `simulation_engine/settings.py`:
```python
MEMORY_N_COUNT = 200  # Increase default memory count
MEMORY_HP = [0.3, 0.5, 0.2]  # Change scoring weights
MEMORY_RECENCY_DECAY = 0.95  # Faster recency decay
```

**To Customize Per-Call:**

All functions accept parameters to override defaults:
```python
# Override defaults for specific call
retrieved = agent.memory_stream.retrieve(
    [anchor], 
    n_count=50,           # Override default
    hp=[0.3, 0.5, 0.2],   # Override default
    time_step=10          # Override default
)
```

### Summary

**Answer:** Yes, memory is **automatically utilized** in every agent interaction. The system:

1. **Automatically retrieves** relevant memories based on the query/question
2. **Scores memories** using recency, relevance, and importance
3. **Selects top N** memories (default: 120)
4. **Appends to prompt** in a specific format alongside the scratchpad
5. **Sends to LLM** as part of the agent description

**Key Points:**
- Memory retrieval happens **automatically** - no manual calls needed
- Memories are **semantically selected** based on query relevance
- Selection uses **three-dimensional scoring** (recency, relevance, importance)
- Default weights: `[0, 1, 0.5]` (recency, relevance, importance)
- Default count: **120 memories** per interaction
- Format: Scratchpad + "==\n" + "Other observations..." + memory contents

---

## Summary

1. **Memory Node Creation:** Only `remember()` and `reflect()` create nodes. Both call internal `_add_node()`.

2. **In-Memory Storage:** Before `save()`, data lives in Python variables:
   - `self.memory_stream.seq_nodes` (list)
   - `self.memory_stream.id_to_node` (dict)
   - `self.memory_stream.embeddings` (dict)

3. **Node Types:** Exactly 2 types:
   - `"observation"` (from `remember()`)
   - `"reflection"` (from `reflect()`)

4. **LLM Interaction:** Single-turn API calls with context injection. No continuous sessions.

---

## Additional Notes

### Memory Retrieval Process

When memories are needed (e.g., for generating responses):

1. **Query Formation:** Current question/topic becomes the "anchor"
2. **Semantic Search:** Embeddings find relevant memories
3. **Scoring:** Combines relevance, recency, and importance
4. **Top-N Selection:** Returns top 120 most relevant memories (default)
5. **Context Building:** Memories are formatted into prompt text

### Memory Persistence

- **In-Memory:** Fast access, volatile (lost on crash)
- **On Disk:** Persistent, loaded on agent initialization
- **Save Frequency:** User-controlled via `agent.save()`

### Best Practices

1. **Call `agent.save()` regularly** to persist changes
2. **Use `remember()`** to store important interactions
3. **Use `reflect()`** periodically to generate insights
4. **Manage time_steps** sequentially for proper recency scoring

