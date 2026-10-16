# Submitting computational tasks on NCSA (Slurm) — general guide

This document explains how to run work on an NCSA-hosted cluster (for example ACCESS Delta) using the **Slurm** workload manager. It is **general**: it does not assume a particular project name, repository path, or software stack. Official **user guide** and **partition** documentation remain authoritative for partition names, account strings, and policy changes.

### How an agent should use this document (single place to return)

1. **Read this guide** for concepts: login vs compute, batch vs interactive, what Slurm directives mean, resource requests, monitoring, and common failures (sections 1–8).
2. **Open that repository’s `NCSA/` folder** (see §12) and read the **project-specific stub** (see §13) for values this guide cannot provide: Slurm **account**, **partition**, filesystem paths, and how this codebase is started on a GPU node.
3. **Submit jobs** using batch scripts and commands that combine site syntax (from official docs) with values from the stub. If something is missing from the stub, **update the stub** after you discover the correct value (§13.4) so the next session still only needs this doc + `NCSA/`.

Without a stub, an agent can follow the *process* but may guess wrong **account** or **partition** and produce rejected jobs or charge the wrong allocation.

---

## 1. What you are doing

- The cluster separates **login nodes** (where you edit files and submit jobs) from **compute nodes** (where heavy work runs).
- **Slurm** decides when your job runs, on which hardware, and for how long, based on what you **request** and what the scheduler can **allocate**.
- You do **not** need to stay connected for a job to run once it is submitted as a **batch job**.

---

## 2. Before you start

- **Allocation:** You need an approved project or allocation that grants CPU or GPU hours on the system you use. Your site documents how to see **which account** to charge (often a short name or a string that includes the system name and resource type).
- **Documentation:** Bookmark the official **user guide** and **running jobs** pages for your system. Partition names, GPU types, and account formats change over time.
- **Environment:** Know how your software runs (e.g. a virtual environment, modules, or a container). Heavy installs are usually done on a login node or in a job; **GPU work** almost always runs on a compute node after you request a GPU.

---

## 3. Interactive work vs batch jobs

| Mode | What it is | When to use it |
|------|------------|----------------|
| **Interactive session** | You open a shell on a compute node (or a GPU node) for a limited time. | Short debugging, one-off commands, checking that the environment sees GPUs. |
| **Batch job** | You submit a **script**; Slurm runs it when resources are free. You can log out. | Long training, sweeps, pipelines, anything that should not depend on your laptop staying connected. |

Batch submission is the standard way to **queue** work so it starts automatically when the scheduler can satisfy your resource request.

---

## 4. How a batch job is defined

A batch job is usually a small **shell script** whose first lines are **directives** that Slurm reads. Those directives are special comment lines (on many systems they begin with a fixed prefix such as `SBATCH`) and they tell Slurm:

- **Which account** to charge.
- **Which partition** (queue) to use — this implies node type, GPU model if any, and policy.
- **How many nodes**, **tasks**, **CPUs per task**, and optionally **GPUs** and **memory**.
- **Maximum wall time** — if the job runs longer, Slurm **terminates** it.
- **Where to write** standard output and standard error (log files).
- Optionally **email** when the job begins, ends, or fails (if your site supports it).

The rest of the script is ordinary shell: change to your project directory, load modules or activate an environment, then run your program.

**Important:** Those directive lines are **not** optional comments from Slurm’s point of view; they are how the scheduler knows what to allocate. Your site’s documentation shows the exact syntax.

---

## 5. What to request (resources and time)

- **Partition:** Choose a partition that matches your need (CPU-only vs GPU, memory, interconnect). Wrong partition names are rejected; **nonexistent** “smaller GPU” partitions are a common mistake — verify names in the current docs.
- **GPUs:** If the partition provides multi-GPU nodes, you typically request a **subset** of GPUs (e.g. two or four on an eight-GPU node). Requesting fewer GPUs can **shorten queue wait** when the scheduler is busy.
- **Time (`--time` or equivalent):** This is a **hard limit**, not a reservation of billed time for the full duration. You are generally charged for **actual** elapsed time up to that limit. Request enough time to finish; if unsure, run a **small test job** and scale up using recorded elapsed time from the accounting tool your site provides.
- **Memory:** Underestimating memory causes the job to fail or be killed; over-requesting can reduce how many jobs fit on a node and may affect scheduling.

---

## 6. Submitting and job identity

- You submit the batch script with the cluster’s **batch submit** command (commonly `sbatch`). The system returns a **job ID**.
- The job moves through states such as **pending** (waiting for resources) and **running**. Your site’s command reference explains how to **list your jobs** and filter by user or job ID.
- **Persistence:** A submitted batch job continues if you disconnect. An **interactive** session usually ends if the connection drops unless you use a terminal multiplexer on the login node; for long work, prefer **batch**.

---

## 7. Monitoring progress and results

- **Queue view:** Use the scheduler’s **queue** command to see state, partition, elapsed time, and sometimes the reason a job is still pending (e.g. waiting for resources, or a policy limit).
- **Start-time estimates:** Some sites show **estimated start** for pending jobs; others do not, especially when the reason is a **quota or billing** limit rather than simple congestion. If start time is “unknown,” read the **reason** field and your allocation documentation.
- **After the job:** Use the **accounting** command (often with the job ID) to see **elapsed time**, **exit state**, and resource usage. This helps you tune future **time** and **memory** requests.
- **Logs:** Standard output and error should be directed to files you can **tail** or open after the run. For multi-step workflows, you may also write application-specific logs inside the job script.

---

## 8. Common issues (conceptual)

- **Job pending with a “billing” or “QOS” style reason:** Your group or user may be over a **fair-share** or **minute** limit; fixing it is administrative or temporal (wait for usage to drop), not something you solve by resubmitting identical large requests repeatedly.
- **Invalid partition or account:** Typos and outdated tutorial examples are common; always confirm current values with `accounts`-style tools and official docs.
- **Job killed at the wall limit:** Increase the time request **only** if policy allows and your workload truly needs it; otherwise split the work or optimize.
- **Zero-second or instant “completion”:** Often the **batch wrapper** exited while the real workload failed immediately; check the **error log** and any logs written by your application.

---

## 9. Practices that generalize well

- Prefer **batch jobs** for anything longer than a quick check.
- Run **small tests** (one GPU, short time, few iterations) before large production jobs.
- Keep **one clear log location** per job ID so you can correlate failures with scheduler records.
- Document for your own project: **account**, **partition**, **typical** CPU, GPU, memory, and time, and the **exact** command sequence from “login” to “submit” so teammates can repeat it without hunting.
- Treat **API keys and tokens** as secrets: use environment variables or secure files, not committed scripts.

---

## 10. What this guide does not replace

- Official **NCSA** or **ACCESS** documentation for your specific system.
- Your **project’s** onboarding (VPN, MFA, storage paths, module names).
- **Application-specific** requirements (containers, conda, distributed launchers, etc.).

Use this document as a **mental model** and checklist. Pair it with that repository’s **`NCSA/`** folder: the **stub** and batch scripts there (§12–13) are the **worked example** with current account and partition strings; site docs supply exact Slurm syntax if your scripts need adjustment.

---

## 11. Summary checklist for “uploading” a task

1. Confirm **account** and **partition** for your allocation (values belong in the repo’s **`NCSA/`** stub once known — §13).  
2. Prepare the **environment** (modules, venv, data paths) on a login node.  
3. Write or reuse a **batch script** under **`NCSA/`** with correct **Slurm directives** and your **run commands**.  
4. **Submit** the script; record the **job ID**.  
5. **Monitor** queue and logs; use **accounting** after completion to tune future requests; **update the stub** if defaults change (**§13.4**).  

Following this flow is the standard, correct way to run substantial work on NCSA GPU (and CPU) resources without tying the run to an interactive terminal session.

---

## 12. Convention: keep NCSA material in a `NCSA/` folder per project

For **each** codebase that runs on NCSA, maintain a top-level directory named **`NCSA/`** in that repository (sibling to things like `src/`, `README.md`, etc.). Use it as the **only** place for cluster-specific artifacts and notes, so agents and humans know where to look.

**Typical contents**

- **Project-specific stub** (§13): one short file with allocation paths, Slurm account, partition, and run recipe (name it clearly, e.g. `DELTA_STUB.md` or `ALLOCATION.md`).
- **Batch scripts** for this project: Slurm job scripts that `cd` to the repo (or to the path on the cluster where the repo lives), activate the environment, and invoke the real command.
- **Optional:** a longer local note (e.g. step-by-step submit commands, log locations) if the stub alone is too small.

**Why a dedicated folder**

- Separates **portable code** from **site-specific** strings and scripts.
- Gives the agent a **predictable path**: always `NCSA/` + stub + scripts.
- Makes it obvious what to copy when you start a **new** repo on the same allocation (copy the folder, then edit the stub).

**Secrets**

- Do not commit API keys, tokens, or passwords inside `NCSA/` (or anywhere). Reference **environment variables** or files outside the repo in the stub and scripts. Use `.gitignore` if you must keep a local-only override file next to the stub.

**Disk layout vs repo root**

- The repo may live under a group path such as **`/projects/<group>/<user>/<reponame>`** on the cluster. The stub should record the **absolute path** used on the cluster for this checkout, not only a relative path, so batch scripts and agents agree on where to `cd`.

---

## 13. Project-specific stub: generate, use, and maintain

This guide is **intentionally generic**. The **stub** is the small, **per-repository** file that holds everything that is **not** generic. The agent should **create** a stub when one is missing, and **edit** it when the cluster or allocation changes.

### 13.1 What the stub must answer

Record at least:

| Topic | Why it matters |
|--------|----------------|
| **Cluster / system name** | Confirms you are on the intended machine (e.g. Delta login host pattern). |
| **Slurm GPU account** | Wrong value → job rejected or wrong allocation charged. Discover with the site’s **account listing** command after SSH (often documented as `accounts` on Delta-style systems). **Filesystem** directory names under `/projects/<slug>/` are **not** proof of the Slurm account string; confirm with that listing or the PI. |
| **Default GPU partition** | Hardware queue; names change. Confirm from official **running jobs / partitions** documentation and live **partition listing** commands. |
| **Default resource profile** | Starting point for GPUs, CPUs per task, memory, and wall time; tune after test jobs. |
| **Repo path on cluster** | Absolute path to this repository’s root on the shared filesystem. |
| **Environment activation** | Modules, conda, venv path, and any required env vars (names only; not secret values). |
| **Command to run** | Exact entrypoint for this project on a GPU node (training script, inference, benchmark, etc.). |
| **Log / output convention** | Where Slurm stdout/stderr go, and any app log dirs. |
| **ACCESS / allocation label (optional)** | Human-readable note (e.g. CIS id) mapping to this Slurm account, if known. |
| **Last verified date** | When account, partition, and paths were last checked on the cluster. |

### 13.2 How to generate a new stub (agent workflow)

1. **SSH** to the cluster login node for the target system (per site instructions: VPN, MFA).
2. **List Slurm accounts** you may use; identify which string corresponds to the **GPU allocation** for this project (PI or admin if ambiguous).
3. **List storage quotas** or inspect **`/projects/...`** / **`/work/...`** paths you can write to; note the **absolute path** where this repo is (or will be) checked out.
4. **Confirm partitions** from documentation and from the scheduler’s partition listing; pick one that matches required GPU type and job size.
5. **Create `NCSA/`** in the repo if it does not exist; add the stub file with the table above filled in, plus a one-line “how to submit” pointer (e.g. which script to run with the batch submit command).
6. **Add a minimal batch script** under `NCSA/` that uses the stub’s account, partition, time, and paths, then runs the documented command after activating the environment.
7. **Run a short test job**; use accounting output to adjust time and memory in the stub and script.

### 13.3 Stub file shape (recommended)

Use plain Markdown so it is easy to edit. Suggested sections: **Overview**, **Slurm**, **Paths**, **Environment**, **Run**, **Submit**, **Maintenance**. Under **Slurm**, repeat the exact strings used in batch directives (account, partition). Under **Submit**, describe from which directory to invoke the batch submit command and which script is canonical.

Keep the stub **short**; move long command transcripts to a separate note in `NCSA/` if needed.

### 13.4 How to maintain the stub

- **After any job rejection** involving account, partition, QOS, or limits: update the stub with the corrected string or policy note.
- **After site announcements** (partition renames, new GPU queues): re-verify partition and docs; update **last verified date**.
- **After moving the repo** on disk: update **absolute path** and any log paths.
- **When switching ACCESS allocation** (different PI or project): replace Slurm account (and paths if they change); note the new mapping in the stub.

Maintaining the stub is what makes **this document alone** sufficient as the **global** instructions: the agent always pairs **§1–11 + §12–13** with the **live** numbers in that repo’s `NCSA/` stub.

---

## 14. Closing checklist (agent)

1. Read **§1–11** in this file for Slurm behavior and pitfalls.  
2. Open **`NCSA/`** and read the **stub**; if missing, create it per **§13.2**.  
3. Use **batch scripts** from `NCSA/` (or create them) with stub values and official syntax from site docs.  
4. Submit, monitor, and **update the stub** when reality diverges (**§13.4**).
