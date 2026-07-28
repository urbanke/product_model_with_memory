# Running on the IPG lab cluster (SLURM)

Head node: `ssh lth.epfl.ch` — the ONLY machine you connect to
(submit jobs from there; never run the experiments' python on it).
It sees the same home dir as all nodes, which is shared storage.

Nodes currently up:

| node   | cores | RAM (allocatable) | select with |
|--------|-------|-------------------|-------------|
| node15 | 64    | ~124 G            | `--constraint=epyc7302` |
| node16 | 20    | 7.6 G (!)         | |
| node17 | 20    | 7.6 G (!)         | |

(!) SLURM's RealMemory for node16/17 is configured as 7,644 MB although
the machines appear to have ~64 G physically — until the admin raises
it, jobs there must request <= ~7 G. All three prepared jobs therefore
target node15; 16/17 remain useful for light jobs (KT re-scores,
spelling variants, small-V runs) with `--mem=6G`.
Jobs exceeding their declared --time or --mem are killed; the limits
set in the job files are generous.

## One-time setup

1. Copy the repo and corpus from your Mac (plain rsync of files
   only — no python involved in the transfer):

       rsync -av --exclude output --exclude .venv \
           ~/Projects/product_model_with_memory/  lth.epfl.ch:~/product_model_with_memory/
       rsync -av ~/Projects/product_model_with_memory/data/text8  lth.epfl.ch:~/product_model_with_memory/data/

2. Install python (one-time, per the admin's instructions): on lth run

       /usr/local/bin/install-anaconda3

   answer the questions, wait, then log out and back in.
   NOTE: the cluster's installer provides an old Anaconda
   (python 3.6.4), which is too old for this code (needs >= 3.7).
   Create a modern environment inside it (one command, on lth):

       conda create -y -n pmm python=3.11 numpy scipy

   Verify with:

       bash ~/product_model_with_memory/cluster/env_setup.sh

   (must print at least one USABLE line; the job files try, in order:
   the pmm conda env, a miniforge pmm env, the system python3, the
   anaconda base.)  If `conda create` fails (e.g. no network route to
   the conda repos), tell Claude the error — plan B is downloading
   Miniforge on the Mac and copying it over.

3. The job files are pre-sized for node15 (mail goes to
   ruediger@gmail.com; change --mail-user if you prefer EPFL mail).
   sf-mid and ct-16k request 32 cores / 48 G each, so they CO-RUN on
   node15 (64 cores); ct-full requests the whole node (64 cores /
   120 G / 7 days) and starts once the others finish. Submit all
   three at once; SLURM handles the ordering.

## Submitting

    cd ~/product_model_with_memory
    sbatch cluster/job_state_family_mid.sbatch     # intermediate-M, full vocab
    sbatch cluster/job_ctree_v16384_d2.sbatch      # context tree V=16,384 D=2
    sbatch cluster/job_ctree_fullvocab_d2.sbatch   # context tree full vocab D=2 (big; start last)

    squeue -u $USER          # status
    tail -f output/<run>/slurm-*.out

Each job is single-node; three jobs use three nodes in parallel.
Caches live under the run's output dir; if home quota is tight, set
CACHE_ROOT in the job files to node-local or shared scratch.

## Getting results back

    rsync -av lth.epfl.ch:~/product_model_with_memory/output/ ~/Projects/product_model_with_memory/output/

Then tell Claude, who reads them via the synced folder.

## Notes

- Assumes home is shared across nodes (standard). If not, tell Claude.
- Jobs are restartable: the table cache persists, so a killed job
  resumed with the same command loses little.
- Multi-node sharding of a single run is NOT yet implemented; it is
  planned for depth-3 / full-vocab runs if a single node is too slow.
