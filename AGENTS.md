# AGENTS.md

This project is finished: the results are in [README.md](README.md), with the predictions and their verdicts in its
section [What I predicted, and what happened](README.md#what-i-predicted-and-what-happened). No human has recomputed
the numbers yet, so there is no `VERIFIED.md`.

## Project context

- A fork of [cywinski/eliciting-secret-knowledge](https://github.com/cywinski/eliciting-secret-knowledge) at upstream
  commit `45aae57` ("add val prompts", 10 Nov 2025). Upstream code can change silently after that point; if something
  does not match the paper, check whether upstream has moved since this hash.
- The paper is Cywiński et al., "Eliciting Secret Knowledge from Language Models",
  [arXiv 2510.01070v2](https://arxiv.org/abs/2510.01070v2); every number here refers to v2.
- The work audits the **User Gender** organism (Gemma 2 9B, `direct` and `standard` prompt sets) with the paper's
  released models, evaluation harness and auditor.

## Roles

- **Dan (Danyil Orel)** is the author. Dan's own text is not edited by agents: the README TL;DR paragraph and Finding 2's
  Confidence text. Propose edits to it instead. Credences are set by Dan; where one is needed, leave
  `<!-- CREDENCE: Dan -->`.
- **Agents** write code, run CPU analyses, check numbers and draft docs to [docs/STANDARD.md](docs/STANDARD.md).

## Environment

- Python in `.venv/`. The CPU environment for numbers and figures is `requirements-figures.txt`; the full research
  environment (GPU) is `requirements.txt`. How to set up, rebuild and run is in [DEVELOPER.md](DEVELOPER.md).
- Raw runs come from the HF dataset `danorel/mats.secret-elicitation-audit` at tag `v1.0`
  (`bash scripts/setup/bootstrap_post.sh --restore-only`).
- **Autonomy window.** `.claude/autonomy/autonomy.py` (copied from bridge-jspace; stdlib only) lifts permission
  prompts until a deadline and restores the previous settings by itself (a detached watcher, plus a SessionStart hook
  if the watcher died). `git push`, `git reset --hard`, `git clean` and `rm -rf` stay denied throughout.
  `make autonomy HOURS=10` (or `UNTIL=00:00`), `make autonomy-status`, `make autonomy-off`, `make test-autonomy`. Do
  not move or rename the repo directory while a window is open: the watcher and the hook hold absolute paths, and a
  moved window would never be revoked. Close it first (`make autonomy-off`).

## Rules for runs and code

- **No training.** All model organisms and evaluation data are the paper's published artifacts.
- **Additive code.** New code goes in new files next to the existing pipeline (a method in `elicitation_methods/`, a
  run script in `user_gender/scripts/`, an analysis in `analysis/`). Do not refactor the upstream harness.
- **A control for every experiment** (a random vector, the base model, a blank input), however cheap the run.
- **Read raw data first.** Read real rollouts, auditor inputs and outputs before trusting an aggregate. If a result
  looks too good, assume a bug or leakage until checked.
- **Every number comes from data by code.** `analysis/` reads raw runs only through `runs.load(<id>)` and writes
  derived files only through `meta.write_derived`; `plots/` reads only `derived/`. No number is typed into a script or
  a doc. Raw runs are registered in `runs.toml` by id; do not edit existing entries without the author's OK.
- **Write derived files on a clean tree** (commit the script first), so `_meta` records the commit and `dirty: false`.
- **Strict figures.** A figure that breaks the styles-plot grammar is not written; fix the figure or the vocabulary.
  Every drawn cell has a row in `plots/status.csv`.

## Rules for writing

- Follow [docs/STANDARD.md](docs/STANDARD.md): the README skeleton and budgets, the finding block, the ledger card,
  the writing rules and the shared vocabulary. `make check` runs the README lint.
- Narrowing a claim is allowed and must be reported; widening one is not. A clean negative result is a valid outcome.
- Keep every existing anchor; when a heading is renamed, add `<a id="<old-anchor>"></a>` where its content now lives.

## Git

- Work on branch `danorel`. Merge into `main`, push, tag or publish (GitHub releases, the HF dataset) only on Dan's OK.
- Never commit `.claude/*.local.md` or `.env`.
