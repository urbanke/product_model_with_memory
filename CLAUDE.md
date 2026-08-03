# Standing instructions for every session in this repository

Ruediger's rules. Follow them without being reminded.

## Language

- Write in plain English. Short sentences. One idea per sentence.
- No jargon. If a technical term is needed, define it the first time.
- Before giving a command to run, say in plain words what it does and
  why. Do the same for every step of a multi-step command.
- Keep answers short.

## Working style

- Say plainly what is measured versus what is inferred.
- One command at a time. Ruediger runs it and pastes the output back.
- Fast, cheap checks before long or expensive runs.
- Never edit a file that a running job has imported. Spawned workers
  re-import files from disk. This has crashed runs before (2 Aug 2026).
- Correctness first, then efficiency.
- Read HANDOVER.md at the start of a session and append an entry when
  something notable is done, decided, or measured.

## The science

- Everything Bayesian: mixtures with priors, integrated exactly by
  algorithms. No learned or tuned weights.
- Small-vocabulary runs (for example V=1024) are behavioral: they rank
  methods in a controlled arena. They are not compression schemes for
  the original file. Real-scheme claims live at the full vocabulary.
