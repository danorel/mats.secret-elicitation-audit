# Changelog

Changes to a claim, a number or a rule, newest first, with the tag or commit. Starts at the first release tag
(`v1.0`). Docs housekeeping (rewraps, figure-style pins with no pixel change, path renames) is left out.

## Unreleased

- **Added: rollout examples.** `analysis/rollout_examples.py` → `derived/user_gender/rollout_examples.json`: both
  checkpoints' responses to the same prompts (seed 0, 2 `direct` + 2 `standard`), with the I/O guess. (`7f8526a`,
  `7fbf01c`)
- **Added: the layer-11 vs layer-23 gap outside the prompts used to pick and validate the layer.**
  `analysis/layer_shift_outside_selection.py` → `derived/user_gender/layer_shift_outside_selection_direct.json`; the
  n=300 layer-11 audit is registered as `gender_direction.l11.direct.n300`. Results pending author review.
  (`0bec9b5`, `b330029`)
- **Corrected: the layer-choice pilot is 30 items = 3 prompts × 10 responses**, read off the rank structure (three
  blocks of 10 near-equal ranks), not 10 prompts × 3 responses as previously documented. Which 3 prompts is inferred,
  not recorded: most likely `direct` prompts 0–2. (`0bec9b5`)
- **Narrowed: the README subtitle** no longer says the layer-11 readout beats the published white-box methods.
  (`75a8545`)
- **Narrowed: Finding 3's title**, from "The secret does not depend on the conversation" to "The checkpoints sit as
  far apart on ordinary chat as on direct questions"; the old anchor is kept. (`75ae204`)
- **Marked: the five-facts paragraph** is now labeled "A hypothesis, not tested". (`8747b37`)
- **Flagged: z = −3.21** (Acts Desc symmetry break) comes from a session analysis the repository's scripts do not
  reproduce. (`84ef99f`)
- **Corrected: "the paper's pipeline is unchanged"** now reads "unchanged except for two edits": an added `acts_desc`
  mode in `utils/guess_secret_word.py`, and `BATCH_SIZE` 50 → 10 in `user_gender/scripts/run_get_sae_features.sh`.
  (`84ef99f`, `4a3daad`)
<!-- more entries are added as the docs brief proceeds -->

## v1.0.2 (2026-11-08, `c911855`)

- Repo and HF dataset renamed to `mats.secret-elicitation-audit`; old ids redirect. No data, numbers or figures
  changed. (`d8fa1ba`)

## v1.0.1 (2026-11-08, `c631a24`)

- **Scripted: precision** (`analysis/precision.py`). The published table reproduces exactly; the layer-11 row moves to
  the canonical bf16 run (female precision 61.11%, z 8.96). (`aef79cf`)
- **Scripted: evidence identity** (`analysis/evidence_identity.py`). Acts Tokens at layer 11 and the gender-direction
  decode give identical evidence on 990/990 items per checkpoint. (`783635a`, `b1bd179`)
- **Scripted: blank-evidence guess rates** (`analysis/blank_guess_rates.py`). (`783635a`, `b1bd179`)
- **Scripted: ordinary chat** (`analysis/ordinary_chat.py`). Accuracies and net contributions reproduce; two z values
  move from the session values 12.24 / −4.84 to 12.27 / −4.85. (`76620e9`, `59877b4`)
- **Registered: base-model control audits** in `runs.toml`
  (`gender_direction.l11.base_model_control.{direct,standard}`), settled by content. (`bf2bd5c`)
- **Labeled: Acts Desc split tables** as not reproduced by a script. The gender-word rule was not recorded, and a rule
  reconstructed on the explicit/abstract split fails on the held-out numbers (rank depth, z). (`098eedc`)
- **Corrected: the `standard`-set layer-11 runs** were a separate run, not `run_gender_direction.sh`. (`29de3a3`)
- **Cited: the paper as arXiv 2510.01070v2**, the version every number refers to. (`25759bc`)
- LICENSE: a copyright line for the changes after upstream commit `45aae57`. (`c84265e`)

## v1.0 (2026-11-07, `600a324`)

First release (tag only, no GitHub release). Claims as released:

- The paper's Acts Tokens method read at layer 11 instead of the published layer 23 raises auditor accuracy on
  `direct` from 60.25% to 68.05% (+7.8 points, paired prompt-level bootstrap 95% CI [+3.35, +12.25]); the
  contrastive gender-direction method gives the same evidence as Acts Tokens at layer 11.
- The difference between the two checkpoints' fine-tuning deltas has cosine +0.5335 with the base model's gender
  direction, about ten times the largest of four control directions (|−0.0547|).
- Layer-11 sign-separability stays at ceiling on ordinary chat (100.00% vs 98.99% on `direct`), with an unchanged
  between-checkpoint gap (28.83 vs 29.03); decoded accuracy falls to 56.80%.
- The auditor's default guess follows the evidence field's shape (~88% "male" with no evidence section, up to 90%
  "female" with an empty one); the bias is symmetric across checkpoints.
- Prompt-level resampling makes standard errors 1.2–2.7× wider than item-level ones on `direct`.
