# Survey patch job: flow and commands

This document explains how the batch job works, where GPU allocation is requested, and **every command you need to run** on your side.

---

## 1. How the job runs when GPU is allocated

- You do **not** request a GPU from inside the job script. You **submit** the job with `sbatch`. Slurm (the scheduler) then:
  1. Puts your job in the **queue**.
  2. When 4 GPUs (and the other resources you asked for) are **free**, Slurm **allocates** them to your job.
  3. Slurm **starts** your script on a compute node that has those GPUs.
  4. Your script runs: it activates the venv and runs the survey patch (four populations in parallel on 4 GPUs).
  5. When the script finishes (or hits the time limit), the job ends and Slurm may send you an email (if you set `--mail-user`).

So: **“Run when GPU is allocated”** means “Slurm starts the script only after it has allocated the requested GPUs.” You don’t have to stay logged in or run anything else after `sbatch`.

---

## 2. Where we request GPU allocation

The **request is in the Slurm script**, not in Python or shell logic. In `NCSA/run_survey_four_populations_patch.slurm` the **#SBATCH** lines tell Slurm what to allocate:

| Line | Meaning |
|------|--------|
| `#SBATCH --account=bdks-delta-gpu` | Which project is charged (your GPU account). |
| `#SBATCH --partition=gpuH200x8` | Which queue/partition (H200 nodes with 8 GPUs each). |
| `#SBATCH --gpus=4` | **Request 4 GPUs.** |
| `#SBATCH --time=02:00:00` | **Request 2 hours**; job is killed after that. |
| `#SBATCH --mem=300g` | Request 300 GB RAM. |
| `#SBATCH --nodes=1` | One node. |
| `#SBATCH --cpus-per-task=40` | 40 CPUs. |

When you run `sbatch NCSA/run_survey_four_populations_patch.slurm`, Slurm reads these lines and queues a job that will get **4 GPUs for 2 hours** (and the rest) when they’re available. The **commands that actually run** (venv, Python, etc.) are the normal shell commands **below** the #SBATCH block; they run **after** allocation.

---

## 3. Every command you need to run (your side)

Do these in order. Everything after step 2 is optional (checking status and results).

### Step 1: One-time setup (if not already done)

On the **login node** (no GPU needed):

```bash
cd /projects/bdks/yueshen7/repos/genagents
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

(And run `python download_models.py` if models aren’t under `/projects/bdks/yueshen7/models` yet.)

### Step 2: Set your email and submit the job

1. **Set your email in the Slurm script** (required for notification):

   Open `NCSA/run_survey_four_populations_patch.slurm` and replace:

   ```text
   #SBATCH --mail-user=YOUR_NETID@illinois.edu
   ```

   with your real email (e.g. `yueshen7@illinois.edu`).

2. **Submit the job from the repo root** (this is the only command that “starts” the run and requests the 4 GPUs):

   ```bash
   cd /projects/bdks/yueshen7/repos/genagents
   sbatch NCSA/run_survey_four_populations_patch.slurm
   ```

   Slurm will print something like:

   ```text
   Submitted batch job 12345678
   ```

   That’s it. The job is queued; it will **run automatically** when 4 GPUs are allocated. You can log out.

### Step 3 (optional): Check queue status

```bash
squeue -u $USER
```

- `PD` = pending (waiting for GPUs).
- `R` = running (your script is executing).

### Step 4 (optional): After the job finishes

- You should get an **email** at the address you set (job end or failure).
- **Slurm output:**  
  - Stdout: `agent_bank/scripts-agent-filtering/survey_logs/slurm-JOBID.out`  
  - Stderr: `agent_bank/scripts-agent-filtering/survey_logs/slurm-JOBID.err`
- **Per-population logs:**  
  `agent_bank/scripts-agent-filtering/survey_logs/survey_patch_mistral_*.log` (and `_llama_`, `_7b_`, `_14b_`).
- **Runtime:**  
  ```bash
  sacct -j JOBID --format=JobID,Elapsed,MaxRSS
  ```
  (Replace `JOBID` with the number from step 2.)

---

## 4. Summary: what runs where

| Who | What |
|-----|------|
| **You** | Run `sbatch NCSA/run_survey_four_populations_patch.slurm` once (after setting email). |
| **Slurm** | Queues the job, allocates 4 GPUs when free, starts the script on a compute node. |
| **Script** | Activates venv, runs four `Surveys/run_survey_patch_attempts.py` processes in parallel (one per GPU via `CUDA_VISIBLE_DEVICES` and `--model-choice`). |
| **Slurm** | Sends you an email when the job ends or fails (if `--mail-user` is set). |

The **only** place GPU allocation is requested is in the **#SBATCH** lines of `NCSA/run_survey_four_populations_patch.slurm`; `sbatch` sends that request to Slurm.

---

## 5. Why `#SBATCH` lines are not comments (ready-to-use script)

In Slurm, any line that starts with **`#SBATCH`** (with that exact spelling) is a **directive**: `sbatch` reads and uses it when you submit the script. So:

- **Do not** change `#SBATCH` to `##SBATCH` or remove them — then Slurm would ignore those lines and the job would not get the right account, GPUs, time, or email.
- Normal shell comments are lines like `# Submit from repo root...` (no `SBATCH`). Those are ignored by both the shell and Slurm.

With your email filled in (`--mail-user=yueshen7@illinois.edu`), the script is **ready to use**. Run from repo root:

```bash
sbatch NCSA/run_survey_four_populations_patch.slurm
```

---

## 6. Do you need tmux for the job to persist?

**No.** With **sbatch**:

- You run `sbatch` on the **login node**. Slurm **queues** the job and returns immediately.
- The job **runs on a compute node** when Slurm allocates resources, **independently of your SSH session**. If you disconnect right after submitting, the job still runs.
- Output goes to files (`slurm-JOBID.out`, survey logs); you get email when it ends.

So the Slurm job and your request **do not** depend on your connection. You do **not** need tmux (or screen) for the job to persist.

Use **tmux** (or `screen`) when you want an **interactive** session that survives disconnect, for example:

- You use **srun --pty /bin/bash** to get a shell on a compute node and want that shell to stay alive if you disconnect, or  
- You run long interactive commands on the login node and want to reattach later.

For **submitting** a batch job, running `sbatch` once is enough; after you see “Submitted batch job …”, you can safely log out.

---

## 7. What RESERV and FEATURES mean in `squeue`

- **RESERV:** Shows an **advanced reservation** ID if your job is part of a reservation (e.g. a guaranteed time slot). Usually **(null)** for normal jobs.
- **FEATURES:** Shows **constraints/features** requested or assigned (e.g. GPU type, interconnect). **(null)** means none requested.

You can ignore both when they are (null).

---

## 8. How to check detailed logs (which agent, which attempt)

The main Slurm output (`slurm-JOBID.out`) only shows the wrapper messages. **Per-agent and per-attempt detail** is in the **four per-population logs** (one per GPU):

- `survey_logs/survey_patch_mistral_<PID>.log` — Mistral-Nemo
- `survey_logs/survey_patch_llama_<PID>.log` — Llama-3.1-8B
- `survey_logs/survey_patch_7b_<PID>.log` — Qwen2.5-7B
- `survey_logs/survey_patch_14b_<PID>.log` — Qwen3-14B

`<PID>` is the job's shell process ID (same for all four in one run). From repo root:

```bash
ls -lt agent_bank/scripts-agent-filtering/survey_logs/survey_patch_*.log | head -4
tail -f agent_bank/scripts-agent-filtering/survey_logs/survey_patch_mistral_<PID>.log
```

In those logs you'll see which agent (e.g. `0000`, `0001`), how many attempts are being run, and per-section/question progress.
