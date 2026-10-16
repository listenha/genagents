# genagents — TUX Agent Study

This repository builds the **profile-conditioned LLM agent bank** used in the agent study of
*TUX: Measuring Human–AI Tacit Understanding* (Li, Min, Das Swain & Saha, EMNLP). It extends the
generative-agent architecture of Park et al. (2024), *Generative Agent Simulations of 1,000 People*.

The pipeline:

1. **Seed personas.** 200 GSS-based demographic seed profiles (`scratch.json`) are selected from the
   original agent bank (`agent_bank/scripts-agent-filtering/`).
2. **Persona enrichment.** Each agent answers a 45-question semi-structured life-story interview
   (`Interview/`). Every Q/A turn is stored as a memory node, with periodic reflections. An LLM-written
   holistic profile summary is then added to `scratch.json` (`Agent_Profile_Summary/`).
3. **Task inference.** For every query, the agent retrieves its top-K memories (relevance + importance)
   and answers in persona:
   - **Pre-task survey** (`Surveys/`): 31 items on a 1–5 Likert scale: BFI-10, the BES-A cognitive subscale,
     the REI rational-ability subscale, an SD3 narcissism subset, and a reduced GDMS.
   - **Wavelength game** (`Wavelength_Game/`): 22 spectrum headers, each placed on a 0–100 scale.
4. **Repeated attempts.** Each task is run 3 times per agent and model family, and incomplete attempts
   are replaced, so every agent ends with 3 clean attempts.

The same interview-built memory streams are reused across model families. Only the inference model
changes between populations.

Analysis of the collected data (TUX computation, consistency, trait regressions) is not part of this
repository. The released data bank is at <https://github.com/listenha/tux-data-bank>.

---

## Repository Structure

```
genagents/
├── genagents/                        # Agent architecture
│   ├── genagents.py                  #   GenerativeAgent: load/save, remember, reflect, respond
│   └── modules/
│       ├── memory_stream.py          #   Memory nodes, importance scoring, retrieval, reflection
│       └── interaction.py            #   categorical / numerical / utterance response generation
├── simulation_engine/                # LLM + configuration layer
│   ├── settings.py                   #   ALL configuration (reads machine-specific values from env)
│   ├── gpt_structure.py              #   Provider dispatch (local HF model or OpenAI), prompt filling
│   ├── local_model_adapter.py        #   HF transformers generation + sentence-transformers embeddings
│   ├── llm_json_parser.py            #   Robust JSON extraction from model output
│   ├── agent_selection.py            #   --range / --agents / --all helpers shared by the CLIs
│   ├── global_methods.py             #   File/CSV/JSON helpers
│   └── prompt_template/              #   All prompt templates
├── environment/
│   └── interview/interview_memory_builder.py   # Runs the interview and writes memories/reflections
├── Interview/                        # Persona enrichment (see Interview/README.md)
│   ├── Interview_Script.md           #   Source interview protocol
│   ├── interview_script.json         #   Parsed script used at runtime
│   ├── extract_interview_script.py   #   .md -> .json
│   ├── run_interview.py              #   CLI
│   └── test_interview_builder.py     #   3-question smoke test
├── Agent_Profile_Summary/            # Holistic persona summary -> scratch.json (see its README)
├── Surveys/                          # Pre-task survey (see Surveys/README.md)
│   ├── PRE-TASK SURVEY.md, survey_questions.json, extract_survey_questions.py
│   ├── survey_response_builder.py, run_survey.py
│   ├── run_survey_patch_attempts.py  #   Drop dirty attempts, top up to N clean attempts
│   ├── archive_survey_responses.py
│   └── PROMPT_EXAMPLE.md
├── Wavelength_Game/                  # Wavelength game (see Wavelength_Game/README.md)
│   ├── Game Headers.csv, game_headers.json, extract_game_headers.py
│   ├── wavelength_response_builder.py, run_wavelength.py
│   └── run_wavelength_patch_attempts.py
├── agent_bank/
│   ├── populations/                  # Agent data, one folder per population (see below)
│   ├── scripts-agent-filtering/      # How the 200 seed agents were selected (see README_FILTER.md)
│   └── gss_agents_demographics_*.csv # Demographics of the selected agents
├── main.py                           # Interactive chat with one agent
├── test_agent.py                     # Agent/model loading sanity check
├── download_models.py                # Download model weights into GENAGENTS_MODELS_DIR
├── run_multiple_attempts.sh          # Run survey|wavelength N times over a population
├── .env.example                      # Template for machine-specific configuration
└── requirements.txt
```

### Agent populations

Every population folder holds agents `0000`–`0199`, and all of them share identical `scratch.json` and
`memory_stream/nodes.json`.

| Folder | Inference model (`GENAGENTS_MODEL`) | Survey attempts | Wavelength attempts |
|---|---|---|---|
| `Qwen3-14B_agents` | `14b`, the paper's primary family | 3 | 3 |
| `Qwen2.5-7B_agents` | `7b` | 3 | 3 |
| `Llama-3.1-8B_agents` | `llama3.1-8b` | 3 | 3 |
| `Mistral-Nemo_agents` | `mistral-nemo-2407` | 3 | 3 |
| `gemma-3-12b-it_agents` | `gemma-3-12b-it` | 3 | 3 (one agent has 1) |
| `Ministral-3-14B-Instruct-2512_agents` | `ministral-3-14b-instruct-2512` | 3 | 3 |
| `placeholder_agents` | none: interviewed agents without task data, the template for new families | – | – |

Per-agent files:

```
<population>/<agent_id>/
├── scratch.json                    # Demographics + agent_profile_summary
├── meta.json
├── memory_stream/
│   ├── nodes.json                  # Interview Q/A observations + reflections
│   └── embeddings.json             # Content -> embedding (384-d MiniLM unless re-embedded)
├── survey_responses.json           # {survey_metadata, agent_metadata, attempts[...]}
├── survey_distilled_responses.json # Trait scores derived from survey (paper's four families)
└── wavelength_responses.json       # {game_metadata, agent_metadata, attempts[...]}
```

---

## Installation

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# For GPU inference, install a CUDA build of torch matching your driver, e.g.
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Configuration

Everything that differs between machines is read from environment variables in
`simulation_engine/settings.py`. Nothing server- or path-specific is hard-coded anywhere else, and
every script resolves repo paths from its own location, so any working directory works.

```bash
cp .env.example .env      # edit as needed; .env is git-ignored
set -a; source .env; set +a
```

| Variable | Default | Purpose |
|---|---|---|
| `GENAGENTS_PROVIDER` | `local` | `local` (HF transformers) or `openai` |
| `GENAGENTS_MODEL` | `14b` | Key into `MODEL_REGISTRY` in `settings.py` |
| `GENAGENTS_MODELS_DIR` | unset | Folder of downloaded weights. If unset, or the model's folder is missing, weights load from the HF hub/cache (`HF_HOME`) |
| `GENAGENTS_DEVICE` | `cuda` | `cuda` picks the GPU with most free memory; or `cuda:N`, `cpu`, `mps` |
| `GENAGENTS_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Memory embedding model |
| `GENAGENTS_POPULATIONS_DIR` | `<repo>/agent_bank/populations` | Population root used by `main.py` / `test_agent.py` |
| `GENAGENTS_DEBUG` | `0` | `1` prints every prompt and raw output |
| `HF_TOKEN` | unset | Needed for gated models (Llama, Gemma, Ministral) |
| `OPENAI_API_KEY`, `GENAGENTS_OPENAI_MODEL` | unset, `gpt-5-mini` | OpenAI provider |

To add a model, add an entry to `MODEL_REGISTRY` in `settings.py`. Memory-retrieval hyperparameters
(`MEMORY_N_COUNT=120`, `MEMORY_HP=[0, 1, 0.5]`, …) are experiment settings and also live there.

Optionally pre-download weights:

```bash
export GENAGENTS_MODELS_DIR=/path/to/models
python3 download_models.py                    # the four paper models
python3 download_models.py gemma-3-12b-it     # or specific registry keys
```

---

## Running the Pipeline

Commands are shown from the repo root. The scripts locate their own resources, so only the paths you
pass on the command line depend on the working directory. Use `python3 -u` for live progress.

```bash
# 1. Persona enrichment for NEW seed agents (folders containing only scratch.json).
#    The interview appends memories and does not skip interviewed agents, so never rerun it on an
#    existing population. Profile summaries skip agents that already have one (use --force to redo).
python3 -u Interview/run_interview.py --base path/to/seed_agents --all
python3 -u Agent_Profile_Summary/run_profile_summary.py --base path/to/seed_agents --all

# 2. New model family: copy the interviewed template population
cp -r agent_bank/populations/placeholder_agents agent_bank/populations/MyModel_agents

# 3. Survey and Wavelength, 3 clean attempts per agent
export GENAGENTS_MODEL=14b
python3 -u Surveys/run_survey_patch_attempts.py         --base agent_bank/populations/MyModel_agents --target-attempts 3
python3 -u Wavelength_Game/run_wavelength_patch_attempts.py --base agent_bank/populations/MyModel_agents --target-attempts 3

# Several families in parallel, one GPU each
CUDA_VISIBLE_DEVICES=0 python3 -u Surveys/run_survey_patch_attempts.py --base agent_bank/populations/Qwen3-14B_agents --model-choice 14b &
CUDA_VISIBLE_DEVICES=1 python3 -u Surveys/run_survey_patch_attempts.py --base agent_bank/populations/Qwen2.5-7B_agents --model-choice 7b &
```

The lower-level runners (`run_survey.py`, `run_wavelength.py`) support single agents, ranges, specific
sections/headers, explicit attempt IDs and batch mode. See the module READMEs. To append a fixed number
of attempts without patching, use `./run_multiple_attempts.sh survey|wavelength`; settings are read from
the environment, e.g. `NUM_ATTEMPTS=3 BASE_PATH=... AGENT_RANGE=0000-0049`.

Sanity checks:

```bash
python3 test_agent.py                    # loads an agent and the local model, generates one utterance
python3 main.py [agent_folder]           # chat with an agent (default: Qwen3-14B_agents/0000)
```

---

## Agent Architecture

### Using an agent from Python

```python
from genagents.genagents import GenerativeAgent

agent = GenerativeAgent("agent_bank/populations/Qwen3-14B_agents/0000")

agent.categorical_resp({"Do you enjoy outdoor activities?": ["Yes", "No", "Sometimes"]})
agent.numerical_resp({"How much do you enjoy coding? (1-10)": [1, 10]}, float_resp=False)
agent.utterance([("Interviewer", "Tell me about your favorite hobby.")])
# -> {"responses": [...], "reasonings": [...]} for categorical/numerical; a string for utterance

agent.remember("Participated in a survey about work preferences", time_step=1)
agent.reflect(anchor="work preferences", time_step=2)
agent.save("path/to/new_agent_folder")
```

A new agent starts with `GenerativeAgent()` and `agent.update_scratch({...})`.

### Memory system

- **Nodes**: `observation` nodes come from `remember()` (interview Q/A turns), and `reflection` nodes are
  LLM-generated from retrieved observations (`pointer_id` lists their sources). Each node gets an
  LLM-scored importance (0–100) when created.
- **Retrieval** runs on every query. The query text is embedded and memories are scored by
  `recency·w_r + relevance·w_v + importance·w_i` with `MEMORY_HP = [0, 1, 0.5]` (recency off). The
  top `MEMORY_N_COUNT = 120` are injected after the self description (`scratch.json`).
- **Prompt**: `Self description: {scratch}` + `==` + `Other observations about the subject:` +
  the retrieved memories, followed by the task template from `simulation_engine/prompt_template/`.
- Each call is a single-turn LLM request. No chat session state is kept.
- **Embedding mismatch**: when an agent is loaded with an embedding model whose dimension differs from
  the stored embeddings (for example, provider `openai`), its memories are re-embedded in memory. The
  stored `embeddings.json` is rewritten only with `GenerativeAgent(folder, save_reembedded=True)` or
  when the agent is explicitly `save()`d.

---

## References

- Li, Y., Min, H., Das Swain, V., & Saha, K. *TUX: Measuring Human–AI Tacit Understanding.* EMNLP.
- Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R., Liang, P., &
  Bernstein, M. S. (2024). *Generative Agent Simulations of 1,000 People.*
