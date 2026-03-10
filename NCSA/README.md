# Running genagents on NCSA Delta

This document describes how to set up and run the genagents project on NCSA Delta (ACCESS). The project has been migrated from another server; paths and allocation details are adjusted for NCSA.

## References

- **ACCESS Compute Tutorial** (project root): `ACCESS Compute Tutorial .md` — login, quotas, partitions, `srun`/`sbatch`. *Some info in the doc may be outdated; check [Delta User Guide](https://wiki.ncsa.illinois.edu/display/DSC/Delta+User+Guide) and [NCSA Delta docs](https://ncsa-delta-doc.readthedocs-hosted.com/) for current partition names and account format.*

---

## 1. Login and allocation

- **Login:** `ssh username@login.delta.ncsa.illinois.edu` (use VPN if off-campus).
- **Account:** Your GPU project name is typically `YOUR_PROJECT-delta-gpu` (e.g. `bdks-delta-gpu`). Check with: `accounts`.
- **Interactive GPU session (example):** On Delta, the H200 partition may be **gpuH200x8 only** (no gpuH200x4), so you get a node with 8 GPUs. You can request 2, 4, or 8 GPUs; the node has 8.
  ```bash
  srun -A bdks-delta-gpu --time=04:00:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=40 --partition=gpuH200x8 --gpus=4 --mem=300g --pty /bin/bash
  ```
  Use your actual account (e.g. `bdks-delta-gpu`). **`--time=04:00:00`** means a 4‑hour limit; if your job runs longer, Slurm will kill it. Request enough time for your run (see below).

- **Run when GPU is ready (no need to wait at terminal):** Submit a **batch job** with `sbatch` instead of `srun --pty`. The job is queued and **starts automatically** when resources are allocated. You can optionally get email when the job ends. See **§1b** below for an example script.

### Queue wait and resource tips

- **Estimate wait time:** Use `squeue -u $USER` to see your jobs; queue position and wait depend on partition demand. There is no single command that gives an exact ETA; the scheduler allocates when resources free up.
- **Shorter wait often:** Requesting **fewer** GPUs (e.g. `--gpus=2` or `--gpus=4`) can help. For this project, **4 GPUs** are enough to run the four populations (one model per population).
- **Check available GPU partitions:** See [Delta production partitions](https://ncsa-delta-doc.readthedocs-hosted.com/en/latest/user_guide/running_jobs.html#delta-production-partitions-queues). Run `sinfo -p gpuH200x8` to see node states; `idle` means capacity is available.

### Using 4 GPUs in parallel

The **Slurm script** `NCSA/run_survey_four_populations_patch.slurm` requests **4 GPUs** and runs **four populations in parallel** (one process per GPU via `CUDA_VISIBLE_DEVICES` and `--model-choice`). So you get roughly one quarter of the wall‑clock time compared to running them one after another. The shell script `run_survey_four_populations_patch.sh` is sequential (one population at a time); use the **.slurm** script for parallel execution.

### Job time limit (`--time`) and run duration

- **`--time=01:00:00`** means a **1‑hour** limit. Slurm **kills the job** when that time is reached.
- **Choosing time:** Estimate from a test (e.g. one population × agents × time per agent), or request a **generous limit** (e.g. `04:00:00` or `08:00:00`) so the run finishes before the limit. You are charged only for time actually used, not the full reserved time.
- After a run, check actual runtime: `sacct -j JOBID --format=JobID,Elapsed,MaxRSS` to see how long it took and adjust future `--time` requests.

### 1b. Submit a batch job (run when GPU is allocated, email notification)

Submit the job with **sbatch** so it runs automatically when resources are available. You don’t need to stay logged in or wait at the terminal.

**Full flow and every command you need:** see **[NCSA/JOB_FLOW_AND_COMMANDS.md](JOB_FLOW_AND_COMMANDS.md)** for step-by-step explanation (where GPU is requested, how the job runs when allocated, and the exact commands to run).

1. **Set your email** in `NCSA/run_survey_four_populations_patch.slurm`: replace `YOUR_NETID@illinois.edu` in the `#SBATCH --mail-user=` line so you get an email when the job ends or fails.

2. **Submit from repo root** (this queues the job and requests 4 GPUs for 2 hours; the job runs automatically when allocated):
   ```bash
   cd /projects/bdks/yueshen7/repos/genagents
   sbatch NCSA/run_survey_four_populations_patch.slurm
   ```
   Check status: `squeue -u $USER`. The script runs **four populations in parallel** on 4 GPUs.

3. **After the job finishes:** You’ll get an email (if set). Inspect `agent_bank/scripts-agent-filtering/survey_logs/slurm-JOBID.out` and the per-population logs there.

The same idea applies to the Wavelength patch: you can copy the Slurm script and replace the survey command with the wavelength patch, or add a second job script.

---

## 2. Model download (no GPU required)

You can **download models on the login node** (or any node) without requesting a GPU. Downloads use the Hugging Face API and do not need CUDA.

```bash
cd /projects/bdks/yueshen7/repos/genagents
source venv/bin/activate
export HF_TOKEN="..."   # if needed for gated models (e.g. Llama)
python download_models.py
```

Run this before requesting a GPU session so models are ready when you start surveys or games.

---

## 3. Project paths (NCSA)

| Purpose | Path |
|--------|------|
| Repo | `/projects/bdks/yueshen7/repos/genagents` |
| Models (download & load) | `/projects/bdks/yueshen7/models` |

These are already set in:
- `download_models.py` → `BASE_MODELS_DIR = /projects/bdks/yueshen7/models`
- `simulation_engine/settings.py` → `MODEL_PATHS` use `_MODELS_BASE = /projects/bdks/yueshen7/models`

---

## 4. Environment setup (venv + requirements)

From the repo root:

```bash
cd /projects/bdks/yueshen7/repos/genagents

# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies (GPU recommended for torch)
pip install --upgrade pip
pip install -r requirements.txt
```

- For gated Hugging Face models (e.g. Llama), set `HF_TOKEN` before downloading or running:
  ```bash
  export HF_TOKEN="your_huggingface_token"
  ```

---

## 5. Download models (before first survey run)

Models must be present under `/projects/bdks/yueshen7/models`. Use the project script (run from repo root, with venv active):

```bash
cd /projects/bdks/yueshen7/repos/genagents
source venv/bin/activate
export HF_TOKEN="..."   # if needed for gated models

python download_models.py
```

This downloads (by default) at least:
- `meta-llama/Llama-3.1-8B-Instruct` → `.../models/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-Nemo-Instruct-2407` → `.../models/Mistral-Nemo-Instruct-2407`
- `Qwen/Qwen2.5-7B-Instruct` → `.../models/Qwen2.5-7B-Instruct`
- `Qwen/Qwen3-14B` → `.../models/Qwen3-14B`

Ensure enough disk space under `/projects/bdks/yueshen7` (and/or quota). Check with `quota` on the login node.

---

## 6. Survey task — overview

The **survey** administers the PRE-TASK SURVEY to agents and writes structured JSON under each agent folder. Survey interactions are **not** recorded to the memory stream.

- **Entry script:** `Surveys/run_survey.py` (run from repo root).
- **Population folders (four base models):**
  - `agent_bank/populations/Llama-3.1-8B_agents`
  - `agent_bank/populations/Mistral-Nemo_agents`
  - `agent_bank/populations/Qwen2.5-7B_agents`
  - `agent_bank/populations/Qwen3-14B_agents`

For each population, the code uses the **matching** base model; the active model is selected via `simulation_engine/settings.py` → `MODEL_CHOICE`. Surveys and games run **one question / one header per inference** (no batch) for reliability.

---

## 7. Run survey for the four agent populations

**Prerequisites:** venv activated, models already downloaded under `/projects/bdks/yueshen7/models`.

**Patch to N attempts (recommended):** If some agents already have one or more survey attempts and you want to **patch up to a target number of attempts** without overwriting valid ones (and to clear dirty attempts that have missing/empty responses), use the NCSA patch script. It runs **one question per inference** (no batch).

From repo root, set the target number of attempts (default 3) and run:

```bash
cd /projects/bdks/yueshen7/repos/genagents
source venv/bin/activate

# Tunable: target attempts per agent (edit the script or pass when we add env var)
export TARGET_ATTEMPTS=3
./NCSA/run_survey_four_populations_patch.sh
```

The script, for each of the four populations: sets `MODEL_CHOICE`, then for each agent loads `survey_responses.json`, removes any **dirty** attempts (attempts with missing or empty response entries), and runs new survey attempts until the agent has `TARGET_ATTEMPTS` clean attempts. Do **not** use `--batch-by-section`; the script uses single-question mode.

**One-shot run (all agents, one attempt each):** To run a single full pass without patching:

```bash
./agent_bank/scripts-agent-filtering/run_full_survey_all_models.sh
```

Do not pass `--batch-by-section` so that each question is run per inference.

Survey outputs (e.g. `survey_responses.json`) are written under each agent directory in the corresponding population folder.

---

## 7b. Wavelength game — patch to N attempts (four populations)

To patch **Wavelength game** attempts for the four populations (clean dirty attempts, then run until each agent has `TARGET_ATTEMPTS`), use one header per inference (no batch):

```bash
cd /projects/bdks/yueshen7/repos/genagents
source venv/bin/activate
export TARGET_ATTEMPTS=3
./NCSA/run_wavelength_four_populations_patch.sh
```

The script calls `Wavelength_Game/run_wavelength_patch_attempts.py` for each population; it does **not** use `--batch-by-header`.

---

## 8. Matters to watch

- **Account and partition:** Use your actual allocation (e.g. `bdks-delta-gpu`) and a valid partition (e.g. `gpuH200x8` or whatever is current on Delta). The tutorial doc may show older partition names (e.g. `gpuA100x4`).
- **Disk:** Models and outputs live under `/projects/bdks/yueshen7`. Stay within project and user quotas (`quota`).
- **Gated models:** Llama (and possibly others) need `HF_TOKEN` for download and sometimes for loading. Set `HF_TOKEN` in the environment when running `download_models.py` and when running surveys if the loader uses the token.
- **GPU:** Surveys run faster with GPU. Start an interactive GPU job (e.g. with the `srun` example above) or submit a batch job that runs the survey script(s) inside the same environment (venv, same paths).
- **Long runs:** For full populations, run inside `tmux` or `screen`, or submit via `sbatch` so the job continues if the SSH session drops.
- **Survey options:** `Surveys/run_survey.py` supports `--no-reasoning`, `--batch-by-section`, `--range`, `--agents`, etc. See `python3 Surveys/run_survey.py --help`.

---

## 9. Quick checklist (survey + wavelength)

1. Log in to Delta. **Download models on the login node** (no GPU): `cd .../genagents`, `source venv/bin/activate`, `python download_models.py`.
2. Start an interactive GPU session (e.g. `srun -A bdks-delta-gpu ... --gpus=4 ... --pty /bin/bash`).
3. `cd /projects/bdks/yueshen7/repos/genagents`, activate venv, `pip install -r requirements.txt` if not already done.
4. (Optional) `export HF_TOKEN="..."` for gated models.
5. **Survey patch (four populations, N attempts):** `export TARGET_ATTEMPTS=3` then `./NCSA/run_survey_four_populations_patch.sh`.
6. **Wavelength patch (four populations, N attempts):** `export TARGET_ATTEMPTS=3` then `./NCSA/run_wavelength_four_populations_patch.sh`.
7. Check logs in `agent_bank/scripts-agent-filtering/survey_logs/` and `wavelength_logs/`; check agent folders for `survey_responses.json` and `wavelength_responses.json`.
