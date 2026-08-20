# CLAUDE.md

## Project context

This repo is a fork of `cywinski/eliciting-secret-knowledge` (arXiv 2510.01070v2,
"Eliciting Secret Knowledge from Language Models", Cywiński, Ryd, Wang,
Rajamanoharan, Nanda, Conmy, Marks). This is work for a MATS 12.0 application to
Neel Nanda's stream — a ~20-hour scoped research exercise, not a production project.

Forked at upstream commit `45aae57` (`cywinski/eliciting-secret-knowledge`, "add val
prompts", 2025-11-10) — the last upstream commit before this fork's own history
starts. Upstream code can change silently after this point; if something doesn't
match the paper's description, check whether upstream has moved since this hash.

Neel's task, verbatim: "Eliciting Latent Knowledge: Can we use interpretability to
elicit secret knowledge from a model? What techniques work best? In Cywiński et al
my scholars taught a model a secret word by training it on descriptions of that
word, and then retrieved it with both black and white box techniques. Can you do
better? (their models)"

We do **not** train any new models. We reuse the paper's published model organisms
and evaluation harness, and try to design a new elicitation method that beats their
published white-box numbers on one benchmark setting. "Do better" is read literally
here: raise a success-rate number, not critique the paper's methodology in the
abstract.

## The benchmark (background you need before touching code)

Three "secret-keeping" model organisms — each is a base model fine-tuned to know
something and use it, but deny it under direct questioning:

- **Taboo** (Gemma 2 9B) — trained to hint at a secret word without saying it.
- **SSC** (Llama 3.3 70B) — follows a Base64-encoded hidden instruction, denies
  decoding it.
- **User Gender** (Gemma 2 9B) — answers consistently with a secret user gender,
  denies knowing it.

Each organism is evaluated on two prompt sets: `standard` (normal chat where the
secret is behaviorally relevant) and `direct` (adversarial "tell me your secret"
prompts). A deliberately weak "auditor" LLM (e.g. Gemma 3 4B) is given evidence
about the secret-keeper — either its raw output, or information extracted from its
internal activations — and has one shot to guess the secret.

**Our target organism is User Gender.** It's the only one where white-box
(interpretability-based) and the best black-box method are within the same order of
magnitude on `direct`: 86.95% (best published white-box, SAE feature descriptions)
vs. 95.45% (prefill attack). Taboo's `direct` set is a degenerate regime — every
method scores ≤7% there. Do not use Taboo/`direct` as a test bed; it isn't
informative.

The repo's pipeline (per its README) has 3 separate steps:
1. Run the secret-keeping model, save its responses.
2. Extract information from internal activations → JSON.
3. An auditor LLM reads that JSON and guesses the secret.

Any new extraction method we build should plug into step (2). Step (3) — the
auditor call — should be reused unmodified wherever possible, so results stay
comparable to the paper's numbers.

## Research hypotheses

The specific hypotheses and technique under test for this application are tracked
privately in `.claude/HYPOTHESES.local.md` (gitignored — this repo is public, and
that file is unpublished application material). Implementation code that comes out
of that plan is still committed normally here; only the hypothesis writeup itself
stays out of git.

## Ground rules

- No training. All model organisms and eval data are the paper's published
  artifacts — we only add new extraction/analysis code.
- **Reproduce the paper's published numbers on User Gender (both `standard` and
  `direct`) before writing any new code.** If a number doesn't reproduce, debug why
  before proceeding — that's part of the work, not something to route around.
- Keep new code additive: new files/modules alongside the existing pipeline,
  following the existing repo's own conventions (e.g. a new extraction method goes
  in `elicitation_methods/` next to `residual_tokens.py`, with a matching
  `user_gender/scripts/run_*.sh` wrapper). Do not refactor the existing harness —
  time spent polishing their code doesn't count toward the actual research
  questions and isn't billable under the application's hour-accounting rules.
- Every experiment needs its own baselines/controls (e.g. a random-vector control,
  a base-model control) — no exceptions, even under time pressure. Skipping a cheap
  control is an explicit rejection criterion for this application.
- A clean negative result with an honest mechanistic explanation is a fully valid
  outcome. Do not overstate marginal or noisy results as positive findings.
- Total time budget for the whole application is ~20 hours (see `log/` for spend
  so far). Stop and write up an honest negative result rather than continuing to
  iterate past budget.
- Read raw data before trusting any aggregate number: actual `direct` rollouts,
  actual auditor inputs/outputs, actual activations. If a result looks too good,
  assume bug or leakage until checked.
- Note the exact commit hash of the upstream repo this was forked from, and record
  it — code can silently change upstream (already bit us once with the paper's
  v1→v2 abstract flip; don't let it happen again with the code).