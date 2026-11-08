# Changelog

Changes to a claim, a number or a rule, newest first, with the tag or commit. Starts at the first release tag
(`v1.0`). Docs housekeeping (rewraps, figure-style pins with no pixel change, path renames) is left out.

## v1.1.1 (2026-11-08)

Docs only; no data, numbers or figures changed.

- **Added: the author's credences** after each finding's Confidence, with what would change each: Finding 1 95%,
  Finding 2 80%, Finding 3 95%, Finding 4 90%.
- **Final: "What I predicted, and what happened"**, read by the author; the draft marker is removed, no words changed.

## v1.1.0 (2026-11-08)

- **Restated: the auditor's "male" share with no evidence section is 88.5%**, printed exactly (it sits on a half, like
  the headline pair), not "about 88%". `readme_numbers.json` (`io_male_guess_direct`).
- **Added: the TL;DR states the gain outside the selection prompts** (6.9 on 70 prompts), and the Status line names
  verification level 2: every README number but the listed exceptions is matched to its source by `make check`.
- **Corrected: the base-model control's female-side share is 97–99%, not 96–99%.** `analysis/readme_numbers.py`
  counts it from the layer-11 projections: 96.97% of items on `direct` and 99.0% on `standard`, the same on both
  checkpoints. Only the README's five-facts paragraph carried the range.
- **Added: `analysis/readme_numbers.py`** → `derived/user_gender/readme_numbers.json`, the source of README numbers no
  other script wrote (the 8.30 and 0.5 gaps, d′, the 6.05pp standard signal, the SE ratios, the reproduction z
  scores, the within-prompt projection range, the surviving anchor pairs, the unparsed grader responses).
- **Narrowed: the 7–13% unparsed grader responses** now name their runs: the blanked token-list and SAE-feature
  calibration runs. The other zero-information runs (I/O, blanked description) are under 1%.
- **Removed: the ~50% reorder share and the per-item gender-word counts** (3.883 vs 3.921 male, 2.046 vs 2.058 female)
  from Finding 1. The float32 evidence file is not a registered run, and session 7's gender-word rule was not recorded;
  the sentences keep the claim (many items reordered, gender-word content barely changed).
- **Added: rollout examples.** `analysis/rollout_examples.py` → `derived/user_gender/rollout_examples.json`: both
  checkpoints' responses to the same prompts (seed 0, 2 `direct` + 2 `standard`), with the I/O guess. (`7f8526a`,
  `7fbf01c`)
- **Added: the layer-11 vs layer-23 gap outside the prompts used to pick and validate the layer.**
  `analysis/layer_shift_outside_selection.py` → `derived/user_gender/layer_shift_outside_selection_direct.json`; the
  n=300 layer-11 audit is registered as `gender_direction.l11.direct.n300`. On the 70 prompts outside both looks the
  gap is +6.86 [+2.29, +11.64]; outside the inferred pilot (97 prompts) +8.14 [+3.81, +12.53]. Descriptive, not held
  out; a ledger card ([outside selection](EXPERIMENTS.md#outside-selection)) holds the detail. (`0bec9b5`, `b330029`)
- **Added: a Read line on the names card.** The post hoc template choice pushed toward a false positive, and the
  eight-name control removed it.
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
- **Restructured: the README** to the docs standard (TL;DR, Setup, four findings with folds, predictions, Reproduce).
  No claim widened; old anchors kept. New: a draft "What I predicted, and what happened" section (three rules, two
  failed). The pilot is described as 3 prompts × 10 responses there too.

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
