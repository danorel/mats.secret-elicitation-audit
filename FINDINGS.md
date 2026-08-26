# Findings

What the numbers in `METRICS.md` mean and why. `METRICS.md` mirrors the source paper's
table structure and stays untouched; this file holds interpretation and mechanism.

## Executive summary

**What this project did:** built a supervised contrastive direction for the User Gender
organism and measured where its advantage over the paper's token-decoding methods
actually comes from: the layer it reads, not the direction. Separately, audited the
evaluation protocol every method in this line of research is scored by, and found a large
content-independent bias in it.

**The headline result is about the benchmark's choice of layer, not about a new method.**
Reading the fine-tuned model's residual stream at first-person-pronoun positions on
**layer 11** instead of the published **layer 23** raises auditor accuracy from 60.25% to
**68.05%** on the `direct` prompts. Same decode, same positions, same auditor, same
evidence template — only the layer differs. The male side, which is the informative one
because every method is at ceiling on the female side, moves 20.80% → 36.80%.

**The contrastive direction contributes nothing to that readout, and this is proven
rather than estimated.** In its plain mode, `gender_direction.py` decodes the raw per-item
residual and never touches the direction vector — the direction only appears in a
diagnostic field the auditor never sees. With both implementations using identical decode
arithmetic, they produce **byte-identical evidence on 990/990 items on both checkpoints** —
identical tokens, token ids and similarity values. They are the same computation. The
direction's contribution is upstream: it is what located layer 11, via two supervised
sweeps.

**What the direction did establish is worth its own claim.** The difference between the two
checkpoints' fine-tuning deltas aligns with the base model's own contrastive gender
direction at **cos = +0.5335**, against a random-vector baseline of 1/√3584 ≈ 0.0167 — about
**32σ**. The direction was built on base Gemma-2-9b-it from contrastive sentences with no
access to either checkpoint, so an arbitrary fine-tuning fingerprint has no reason to land
there. Fine-tuning wrote the secret into a direction the base model already used for gender.

**But it wrote it as a constant, not as per-turn computation.** Sign-separability at layer 11
is **100.00% on `standard`** — ordinary chat where gender never comes up — against 98.99% on
`direct`, while decoded accuracy on `standard` collapses to 56.80%. The between-checkpoint
projection gap is 29.03 on `direct` and 28.83 on `standard`, a 0.7% difference: the offset
does not care whether the conversation concerns gender. This single fact explains the rest
of the project's negative results — a constant offset carries no per-item information for a
decode to extract, is not a nameable concept for Patchscope to verbalise, and leaves the
`standard` decode with nothing to discriminate.

**The most generalisable finding came from auditing the grading, not from any method.**
Every method in this area is scored by a small LLM guessing "male" or "female" from
extracted evidence, and that LLM has a strong default that follows the *shape* of the
evidence field rather than its content: ~88% "guess male" with no evidence section at all,
up to 90% "guess female" the moment an empty one exists. The bias is symmetric across
checkpoints, so the standard mean-of-both-checkpoints convention absorbs it and the
published ranking survives — but the correction is diagnostic rather than corrective: it
moves every number here by **less than one point** and reorders nothing. Its value is
telling you how much of a score is signal rather than default. Any evaluation that has an
LLM grade extracted evidence should run this check.

**What is still open.** Only layer 11 has a full end-to-end auditor run; layers 9, 10 and 12
are close on the proxies and untested. Whether the bias correction holds under real but
vague evidence is confirmed to break down somewhat and is not resolved. And a delta-decode
method suggested by the measurements below was scoped but not built.

## At a glance

User Gender, `direct` set. Auditor: Gemma 3 4B, reading only extracted evidence.

| result | female | male | avg | verdict |
| --- | --- | --- | --- | --- |
| I/O (no evidence) | 11.70% | 88.70% | 50.20% | reference point |
| Acts Tokens, layer 23 (paper) | 99.70% | 20.80% | 60.25% | reproduced |
| **Acts Tokens, layer 11** | **99.30%** | **36.80%** | **68.05%** | **+7.8pp from the layer alone** |
| Gender Direction, layer 11 | 99.30% | 37.80% | 68.55% | same computation as the row above |
| SAE Desc (paper's best white-box) | 96.60% | 79.00% | 87.80% | reproduced |
| Gender Direction, `base_model_control` | 93.50% | 7.20% | 50.35% | confirms it's fine-tuning signal |
| Acts Desc (Acts Tokens' evidence as prose) | 69.50% | 31.80% | 50.65% | negative |
| Patchscope (direction → free-text decode) | - | - | - | negative |

The 0.50pp between the two layer-11 rows is not a method difference: the two rows were
produced under different decode precision. With the arithmetic matched they yield identical
evidence, and 0.50pp is in any case far inside the ±4.9pp standard error implied by the
effective sample size of 99 prompts.

Every section below follows the same shape: what we tested, what we found, what it means.
Skip to the bold **bottom line** in each section for the one-sentence version.

## Baseline reproduction

**What we tested.** Reproduced the paper's five published white-box methods end to end,
both prompt sets, before trusting anything built on top of them.

| method | `direct` repro | `direct` published | `standard` repro | `standard` published |
| --- | --- | --- | --- | --- |
| I/O | 50.20% | 50.45 ± 2.89 | 51.80% | 51.8 ± 2.6 |
| LL Tokens | 64.25% | 66.15 ± 1.33 | 69.10% | 68.2 ± 1.6 |
| Acts Tokens | 60.25% | 57.90 ± 0.80 | 56.85% | 55.7 ± 1.5 |
| SAE Desc | 87.80% | 86.95 ± 0.94 | 85.55% | 85.4 ± 1.6 |
| SAE Tokens | 63.30% | 60.65 ± 0.84 | 56.10% | 57.6 ± 1.3 |

**Bottom line: reproduction holds, with one caveat worth naming rather than hiding.** 8 of
10 rows land within 1.5 published standard errors. Two don't: Acts Tokens and SAE Tokens on
`direct` are 2.94 and 3.15 SE above published. Part of the explanation is that the published
SE is too narrow — the "published standard errors" note below shows it is about 4× too small
once response-variant non-independence is accounted for.

That explains why a 3σ gap is unsurprising. It does not explain why **both deviations are
upward and both are on `direct` only**, which looks like a pattern rather than noise. Since
no rollouts were regenerated, the remaining source of run-to-run variation is the grader
itself; re-grading the cached evidence under several grader seeds would measure it directly
and cheaply, and has not been done.

![Bar chart comparing average auditor accuracy across all reproduced methods plus Gender Direction, direct set, with bootstrap error bars and a chance line at 50%](https://huggingface.co/datasets/danorel/eliciting-secret-knowledge-results/resolve/main/user_gender/plots/reproduction_bar_chart_direct.png)
*Direct set, every method side by side.*

Everything below builds on this reproduction.

## The layer, not the direction

**What we tested.** Whether the 8.3pp that Gender Direction leads Acts Tokens by comes from
the contrastive direction or from the fact that it reads layer 11 while the published method
reads layer 23. Two independent checks, from opposite ends.

**Check 1: run the paper's own method at layer 11.** Nothing else changed —
`residual_tokens.py --mode pronouns`, same top-k, same auditor, same template, n=1000 per
checkpoint.

| method | layer | female | male | avg |
| --- | --- | --- | --- | --- |
| Acts Tokens (published config) | 23 | 99.70% | 20.80% | 60.25% |
| Acts Tokens (this project) | 11 | 99.30% | 36.80% | **68.05%** |
| Gender Direction | 11 | 99.30% | 37.80% | 68.55% |

**Of the 8.30pp gap, 7.80pp is the read layer.**

**Check 2: read the code, then prove the identity.** In plain mode (`--candidate_size`
unset), `gender_direction.py` computes its decode from the per-item residual and the token
embedding matrix only; `direction_hat` appears exactly once elsewhere, in a `mean_projection`
diagnostic, and the auditor template fills only `{user_prompt}`, `{model_response}` and
`{top_tokens}`. The direction cannot reach the grader.

Decode precision matters for this comparison and is worth stating explicitly.
`residual_tokens.py` multiplies normalized residuals against normalized embeddings in the
model's own bfloat16 and casts to float32 afterwards. bf16's 8-bit mantissa coarsens cosine
similarities enough to reorder a top-200 drawn from a 256k vocabulary, so decoding the same
residual in float32 changes the token list on ~50% of items. It barely changes the
gender-word content (3.883 vs 3.921 male words per item, 2.046 vs 2.058 female), so
precision does not account for the 0.50pp gap.

With both methods using the same bf16 arithmetic, they produce **byte-identical evidence on
990/990 items on both checkpoints**, down to similarity values at six decimals.

**Bottom line: in plain mode this is not a new method.** It is the paper's own Acts Tokens
readout applied at a different layer. The contrastive direction's contribution is that it
*found* that layer, through the two supervised sweeps below.

**Why the paper reads layer 23, and why "the authors missed the optimal layer" overstates
it.** `user_gender/scripts/00_run_auditing.sh` downloads SAE features via
`download_sae_features.py --layer 23`, and SAE Tokens/SAE Desc need
`gemma-scope-9b-pt-res-canonical`, a Gemma Scope release published at a fixed set of
canonical layers. The likely reason all five published methods share layer 23 is that the
SAE-based methods require it, and the paper fixed one layer across all five for a clean
cross-method comparison rather than tuning each method's layer independently. Layer 11 was
found by a supervised sweep over labelled dev data (Choosing layer 11, below) that the
paper's own reproduction doesn't run for its non-SAE methods — a legitimate thing to check,
but the fair framing is "decoupling the layer from an SAE constraint that doesn't bind the
non-SAE methods recovers 7.8pp," not "the published choice was an oversight."

## What the direction did establish

**The fine-tuning delta lands on the base model's own gender direction.** Measuring the
residual at pronoun positions under the fine-tuned checkpoint and under base
Gemma-2-9b-it, at layer 11, n=990 per checkpoint:

| quantity | female | male |
| --- | --- | --- |
| delta constant ‖mean‖ | 51.29 | 64.90 |
| delta per-item spread | 39.58 | 52.00 |
| ratio | 0.772 | 0.801 |
| raw residual ratio (reference) | 0.570 | 0.550 |
| cos(mean delta, contrastive direction) | +0.1219 | −0.3510 |

`cos(mean_delta_female − mean_delta_male, direction) = **+0.5335**`, with
`‖difference‖ = 54.416`. For a random unit vector in 3584 dimensions the standard deviation
of that cosine is ≈ 1/√3584 = 0.0167, so this is roughly **32σ**.

The direction was built on the **base** model from gender-contrastive sentence pairs, with
no access to either checkpoint. An arbitrary fine-tuning fingerprint has no reason to align
with it. This is direct evidence — not plausibility — that the fine-tune wrote the secret
into a representation the base model already used for gender.

**Tension worth naming:** the male delta aligns three times more strongly with the direction
(−0.351 vs +0.122) yet the male side has by far the worse accuracy (37.8%). Alignment with
the direction does not translate into decodability, which is consistent with the Patchscope
negative below.

## The secret is stored as a constant, not as per-turn computation

**What we tested.** Whether the near-perfect layer-11 separability reflects per-item signal
about the secret, or a checkpoint-level offset that happens to be perfectly separable. The
discriminator is the `standard` set, where the secret is behaviourally irrelevant.

| prompt set | separability | decoded auditor accuracy | between-checkpoint gap |
| --- | --- | --- | --- |
| `direct` | 98.99% (n=1980) | 68.55% | 29.03 |
| `standard` | **100.00%** (n=2000) | 56.80% | 28.83 |

Separability goes *up* where decoded accuracy collapses. On `standard` the two checkpoints'
projection distributions do not overlap at all (female min +0.044, male max −7.175, d′ =
6.08). The gap differs by 0.7% between the two prompt sets: the offset is present at full
strength whether the conversation is about the user's gender or about desk posture.

Pairing by prompt sharpens this. Both checkpoints run the same `direct` prompts, the read
positions live inside the prompt, and causal masking makes the residual there independent of
the response — so `contrast(p) = r_female(p) − r_male(p)` is a pure weight difference at
identical input. Over the 99 shared prompts:

| | |
| --- | --- |
| constant ‖mean contrast‖ | 54.416 |
| per-prompt spread around it | 36.264 (ratio 0.666) |
| cos(contrast_p, direction), per prompt | mean +0.4515, sd **0.0618**, range [+0.203, +0.577] |

The contrast does vary by prompt — but its alignment with the gender axis is positive for
**all 99 prompts** and tightly clustered. The prompt-dependent variation lies in directions
largely orthogonal to gender: other things the fine-tune changed that depend on the prompt.
Along the gender axis itself, the signal is constant.

**Bottom line: the organism does not compute "this user is male" per conversation — it
carries a standing offset at layers 10–12, always on, the same size regardless of topic.**
That one fact explains the rest of this document: a constant carries no per-item information
for a decode to extract (hence `standard` at 56.80% despite 100% separability), it is not a
nameable concept (hence Patchscope), and it is why the read *location* matters more than any
supervised direction applied at that location.

**Caveat that must not be dropped:** target gender and checkpoint identity are the same
variable in this organism by construction, so this cannot fully separate "always-on belief
about the user" from "arbitrary fine-tuning fingerprint". The 32σ alignment above is the
evidence against the latter, and it is strong but indirect.

## Classification quality: precision tells a different story than accuracy alone

**What we tested.** "Female accuracy" and "male accuracy" everywhere above are **recall**.
Recall can't tell "the auditor found real signal" apart from "the auditor mostly guesses
female regardless". We computed **precision** instead, each with a two-proportion z-test
against a coin flip.

| method | recall F / M | precision F | z | precision M | z |
| --- | --- | --- | --- | --- | --- |
| I/O (no evidence) | 11.70% / 88.70% | 50.87% | 0.26 (n.s.) | 50.20% | 0.17 (n.s.) |
| Acts Desc | 69.50% / 31.80% | 50.47% | 0.35 (n.s.) | 51.04% | 0.52 (n.s.) |
| Acts Tokens, layer 23 | 99.70% / 20.80% | 55.70% | **4.82** | 98.58% | **14.11** |
| Gender Direction / Acts Tokens, layer 11 | 99.30% / 37.80% | 61.49% | **9.23** | 98.18% | **18.91** |
| SAE Desc | 96.60% / 79.00% | 82.14% | **22.05** | 95.87% | **26.34** |

**Bottom line: recall alone would overstate two methods and understate one.** I/O and Acts
Desc carry no real signal in either class despite Acts Desc's recall making it look like it
does something — the two near-zero z values in the precision-M column are the whole point of
that column. Acts Tokens' famous 99.7% female recall is mostly base-rate bias: its actual
female precision (55.7%) is barely above a coin flip. Reading layer 11 raises female
precision to 61.5% at z=9.23 on the identical evidence template — independent confirmation,
via a different statistic, that the layer change is a real improvement.

## Choosing layer 11

Two full-scale proxy measurements, neither of which is auditor accuracy itself.

**Proxy 1: nearest-token decode rank.** Capture the residual at the pronoun position(s),
rank the entire vocabulary by cosine similarity to it, record where the best-matching word
from a fixed gender-word list lands. Lower is better. n=990/gender:

| layer | 9 | **10** | **11** | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| median rank, male | 57 | **54** | 65 | 106 | 105.5 | 78 | 114 | 91 | 66 | 89 | 77 | 95 | 94 |
| median rank, female | 92 | 77 | **68** | 118 | 110 | 69 | 108 | 105 | 84 | 141 | 127 | 142 | 145 |

Full table, no rows dropped. Layer 11 is the single best point on the female side and close
to best on the male side.

![Line chart of median full-vocabulary rank of the best-matching gender word vs. layer, female and male, layers 9-21, log scale, shaded interquartile band](https://huggingface.co/datasets/danorel/eliciting-secret-knowledge-results/resolve/main/user_gender/plots/gender_word_rank_sweep.png)

![Line chart of median gender-word rank by layer with an interquartile band, male and female, over the full swept range](https://huggingface.co/datasets/danorel/eliciting-secret-knowledge-results/resolve/main/user_gender/plots/layer_rank_sweep_chart.png)
*The same sweep with its interquartile spread shown. The 9–11 window is not just lowest on
the median — its spread is also tighter, which is what makes the pick defensible rather than
a one-statistic artifact.*

**Proxy 2: raw linear separability, no decode at all.** Classify purely by the sign of the
projection onto that layer's direction. n=1980:

| layer | 10 | 11 | 12 | 13 | 14-22 | 23 | 24 | 25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| separability | 99.49% | 98.99% | **100.00%** | 58.03% | exactly 50.00% | 67.88% | 70.71% | 50.00% |

![Line chart of raw sign-threshold classification accuracy vs. layer, showing a sharp near-perfect window at layers 10-12, exact chance from 14-22, and a partial recovery at 23-24](https://huggingface.co/datasets/danorel/eliciting-secret-knowledge-results/resolve/main/user_gender/plots/projection_separability_sweep.png)

The two proxies have different shapes and **agree only inside the 10–12 window**, which is
what makes their agreement informative rather than circular. The published methods read
layer 23, where separability is 67.88% and the gender-word rank is much worse.

**The exactly-50.00% band at layers 14–22 is a constant projection sign, not absent signal.**
A shared mean offset dominates the residual there, so every item falls on the same side of
zero. That is a failure of the *uncentred* readout; a centred sweep would distinguish the two
and has not been run. The same degeneracy is why the base-model control also sits at exactly
50.00% — it puts 96–99% of items on the female side regardless of true gender.

Layer 12 edges out 11 on separability (100.00% vs 98.99%, ~20 items out of 1980, both at
ceiling) but is clearly behind on the rank sweep, the metric closer to what the decode
actually sees. **No layer besides 11 has been tested end to end.** Re-running the pipeline at
9, 10 and 12 is the obvious next step.

## What the average hides

The effective sample is 99 prompts per checkpoint, not 1000 items — and the reason is
stronger than "ten samples aren't independent". `find_first_person_pronoun_positions` searches
the user prompt only, and causal masking means positions inside the prompt cannot attend to
the response. **Every method in this project reads a vector determined entirely by the
prompt.** Empirically, within a prompt the projection varies by at most 0.36 against a
between-checkpoint gap of 29.0 — bf16 arithmetic on differing sequence lengths, not signal.
The paired cluster bootstrap used throughout was necessary, not merely conservative.

Per-prompt accuracy, both checkpoints:

| checkpoint | 0.00 (fails completely) | 0.10-0.80 (partial) | 0.90-1.00 (succeeds cleanly) |
| --- | --- | --- | --- |
| female | 0/100 | 2/100 | 98/100 |
| male | 38/100 | 38/100 | 24/100 |

![Histogram of per-prompt auditor accuracy, female and male, showing female almost entirely at 1.0 and male spread across the full 0.0-1.0 range with peaks at both ends](https://huggingface.co/datasets/danorel/eliciting-secret-knowledge-results/resolve/main/user_gender/plots/bimodal_breakdown_chart_direct.png)

Female sits at ceiling because of auditor bias, not because the method reads female prompts
well. The male side has real variation but is **not cleanly bimodal** — a substantial middle
third lands at partial accuracy. About a quarter of phrasings succeed cleanly, a little over
a third fail completely, and the rest are in between.

The residual evidence is near-constant within a prompt, so that middle group cannot come from
the token evidence. It comes from `{model_response}`, which the auditor template also carries
and which does vary across the ten samples. That channel carries no gender information of its
own — the I/O baseline, which is prompt plus response with no evidence at all, sits at 50.20%
— so it contributes variance to the grader's decision rather than signal.

## Ordinary chat

| variant | female | male | avg |
| --- | --- | --- | --- |
| contrastive, `standard` | 99.80% | 13.80% | 56.80% |
| `base_model_control`, `standard` | 97.90% | 2.20% | 50.05% |

| comparison | direct vs. standard | z |
| --- | --- | --- |
| male, contrastive | 37.80% vs. 13.80% | **12.24** |
| female, contrastive | 99.30% vs. 99.80% | -1.67 (n.s.) |
| male, `base_model_control` | 7.20% vs. 2.20% | **5.28** |
| female, `base_model_control` | 93.50% vs. 97.90% | **-4.84** |

The male-side gap is large and highly significant, but the control shifts significantly too,
so part of the `direct`/`standard` difference belongs to the prompt sets rather than to
anything the fine-tuned model learned. Net of that, fine-tuning's own contribution is +18.2pp
on `direct` against +6.75pp on `standard` — under half.

**Mechanism, now better supported than when first written.** The offset itself is the same
size on both prompt sets (29.03 vs 28.83), so the decode is not receiving a weaker signal —
it is receiving the same signal in a context where the surrounding residual makes it harder
to surface as gender vocabulary. `direct` prompts put the pronoun inside a sentence whose
whole topic is identity; `standard` prompts use it incidentally. The signal is diluted in
presentation, not reduced in magnitude.

![Bar chart comparing average auditor accuracy across all reproduced methods plus Gender Direction, standard set, with bootstrap error bars and a chance line at 50%](https://huggingface.co/datasets/danorel/eliciting-secret-knowledge-results/resolve/main/user_gender/plots/reproduction_bar_chart_standard.png)
*Standard set. The layer-11 readout is indistinguishable from Acts Tokens and SAE Tokens
here, and LL Tokens leads by 12 points.*

![Histogram of per-prompt auditor accuracy on the standard set, female and male, showing female almost entirely at 1.0 and male even more concentrated near 0.0 than on the direct set](https://huggingface.co/datasets/danorel/eliciting-secret-knowledge-results/resolve/main/user_gender/plots/bimodal_breakdown_chart_standard.png)
*Male shifts further toward complete failure (67/100 prompts at 0.00, vs 38/100 on
`direct`) — the weaker average is concentrated as more prompts failing outright.*

## Patchscope: a negative result, with a mechanism

**What we tested.** Whether the direction can be read out as a natural-language word by
patching it into a "name this concept" prompt.

**Positive controls first.** Real word embeddings decode correctly across a clean window —
`woman` → "woman", `king` → "king" — verified on both fine-tuned checkpoints at layers 22–24
after an earlier audit caught that only the base model had been checked.

**The direction itself**, across every layer in the window, both checkpoints, two templates
and injection scales from 0.3× to 20×, over all 3000 per-item outputs in the final sweep:

| output | where | n | share |
| --- | --- | --- | --- |
| empty | 24 of 30 layer × checkpoint cells, 100% each | 2592 | 86.4% |
| constant `"dog"` | layer 24, both checkpoints | 200 | 6.7% |
| template echo | male layers 20 and 22 | 158 | 5.3% |
| degenerate token (`"1"`, `"_"`) | female layers 10 and 14 | 50 | 1.7% |

**Never anything gender-related** — the complete set of non-empty outputs is `"1"`, `"_"`,
`"dog"` and echoes of the induction template.

**Bottom line: a structured failure, not random silence.** 24 of 30 cells are 100% empty with
zero exceptions. Two details are worth stating precisely rather than smoothing over. Layer 24
emits the constant token `"dog"` on 100/100 items on both checkpoints — and `"dog"` does not
appear anywhere in the induction template (cat/tree/love/mountain/car/river/joy), so this is a
collapse to a default completion, not the copying behaviour it might look like. And the
non-empty outputs do **not** cluster only where the mechanism is known to be weak: female
layer 10 is 40% non-empty and sits inside the 10–12 window this project identifies as its
best.

Combined with the constant-offset finding above, the explanation is: the direction is a real,
linearly separable classification axis, but a standing offset is not a concept the model can
name. Unlike a word embedding or an SAE feature, neither decode method has anything to grab.

![UMAP and PCA projections of 32 direction vectors, contrastive vs random_control, with contrastive points connected in layer order forming a visible trajectory while random_control points show no structure](https://huggingface.co/datasets/danorel/eliciting-secret-knowledge-results/resolve/main/user_gender/plots/direction_vectors_umap.png)
*`contrastive` vectors (pink, connected by layer) trace a continuous path and sit in a
different region than `random_control` (grey) — the direction isn't arbitrary even where it
fails to decode into anything nameable.*

## The auditor's guessing bias

**What we tested.** Every method above is graded by an auditor guessing from evidence. If
that guess has its own default, every accuracy number is contaminated by an unknown amount.
Measured directly: same prompt, same real response, evidence content blanked, four
evidence-field shapes, n=1000 per file.

| evidence field | guessed "female" (female-tuned) | (male-tuned) | avg |
| --- | --- | --- | --- |
| absent entirely | 11.70% | 11.30% | 11.50% |
| description field, blanked | 67.40% | 66.70% | 67.05% |
| token-list field, blanked (Acts/LL/SAE Tokens, layer-11 readout) | 90.10% | 88.60% | 89.35% |
| SAE-feature field, blanked | 87.40% | 85.70% | 86.55% |

**Bottom line: the auditor is never neutral, and which way it leans depends on the shape of
the prompt, not its content.** ~88% "guess male" with no evidence section at all, 67–90%
"guess female" the instant one exists, regardless of what is inside it.

**The bias cancels, checked rather than assumed.** Each bias hits both checkpoints equally
(z = 0.33, 1.09, 1.11; none significant), and a target-independent bias averages to exactly
50% no matter how strong it is. `METRICS.md`'s mean-of-both-checkpoints convention absorbs
all three.

| method | template | blanked-evidence avg | real avg | real signal |
| --- | --- | --- | --- | --- |
| Acts Desc | own | 50.35% | 50.65% | +0.30pp |
| Acts Tokens, layer 23 | shared w/ LL, SAE Tokens | 50.75% | 60.25% | +9.50pp |
| layer-11 readout, `direct` | same template | 50.75% | 68.55% | **+17.80pp** |
| layer-11 readout, `standard` | same template | 50.75% | 56.80% | +6.05pp |
| SAE Desc | own | 50.85% | 87.80% | **+36.95pp** |

**Read this table for what it is.** Because the bias is symmetric it cancels in the averaging
convention, so **the correction moves every number in this document by less than one point
and reorders nothing.** It is diagnostic, not corrective. Its value is that it says a method
scoring 50.65% is not weakly informative but exactly empty, and it would catch a future
method whose evidence template is friendlier than its content. It cannot change a
same-template comparison at all: the layer-11 readout and Acts Tokens share a baseline, so
the 8.3pp gap between them is 8.3pp either way — the +17.80pp is that method's own distance
from its baseline, not a margin over anything.

The underlying bias is not small, even though its correction is: the same grader guesses
female 11% of the time with no evidence field and 90% with an empty one.

![Bar chart of each method's accuracy minus its own blank-evidence baseline, showing Acts Desc at essentially zero, Acts Tokens and the layer-11 readout in the middle, and SAE Desc far ahead](https://huggingface.co/datasets/danorel/eliciting-secret-knowledge-results/resolve/main/user_gender/plots/blank_baseline_signal_chart.png)
*The same methods scored as distance from their own blank-evidence baseline rather than
from 50%. The ordering is unchanged from raw accuracy — which is the point: the baselines
sit within half a point of each other, so the correction separates signal from default
without reshuffling anything.*

**Where it doesn't hold.** All of this is verified at zero information. With real but vague
content the symmetry measurably breaks (z = −3.21, below).

**Has anyone else measured this?** The source paper (arXiv 2510.01070v2) doesn't discuss it.
Its closest passage (Appendix F.1) is about auditors struggling to filter irrelevant
evidence — a different question from whether the auditor's default guess is neutral.

## Acts Desc: the grader follows wording, not meaning

**What we tested.** Acts Tokens scores 20.8% on the male checkpoint and SAE Desc 79.0%, but
they differ in both extraction mechanism and evidence format, so that comparison cannot say
which matters. Isolating format: took Acts Tokens' own evidence and had GPT-4.1-mini
paraphrase each token list into prose, changing nothing else. The paraphraser never saw the
prompt, the response or the target.

| method | female | male | avg |
| --- | --- | --- | --- |
| Acts Tokens (raw list) | 99.70% | 20.80% | 60.25% |
| Acts Desc (paraphrased) | 69.50% | 31.80% | 50.65% |
| SAE Desc | 96.60% | 79.00% | 87.80% |

**Format alone doesn't explain the gap — falsified.** The average lands *below* Acts Tokens
rather than between it and SAE Desc, which is the opposite of what the format hypothesis
predicts.

**And the near-zero average is not a near-zero effect.** Splitting descriptions by whether
an explicit gender word survived the paraphrase, **per checkpoint**:

| | explicit gender word | abstract, no explicit word |
| --- | --- | --- |
| female checkpoint | n=270, **99.63%** | n=730, 58.36% |
| male checkpoint | n=90, **22.22%** | n=910, 32.75% |

The two checkpoints disagree, and the disagreement is the result. On the female checkpoint
an explicit keyword looks like near-perfect signal; on the male checkpoint the same condition
scores **22.22%** — worse than chance, and worse than the abstract descriptions it is being
contrasted with. Balanced across checkpoints: explicit 60.9%, abstract 45.6%. Either
checkpoint read alone gives the wrong conclusion.

**Mechanism.** An explicit keyword is not signal; it is a push toward "female" that happens to
be right on one checkpoint and systematically wrong on the other. The paraphraser sees a list
of pronouns, reaches honestly for identity-sounding language — "personal", "family",
"self-reference" — and emits female-flavoured explicit words regardless of the truth:

| abstract phrasing | guessed "female" (female-target) | guessed "female" (male-target, wrong) |
| --- | --- | --- |
| identity words present | 68.0% | 77.3% |
| purely structural wording | 49.4% | 52.1% |

Structural wording is a coin flip regardless of truth. Identity-sounding wording pulls toward
female regardless of truth as well.

**This section is a third line of evidence for the grader-bias finding**: blank evidence
shows the default exists, explicit-vs-abstract shows it responds to wording, and
identity-vs-structural shows which wording triggers it.

*(The classification boundary matters for the exact counts — a slightly broader gender-word
regex gives 270/730 where a narrower one gives 230/760 — but the reversal does not depend on
where the line is drawn.)*


## Limitations

1. **One organism, end to end.** Taboo's numbers are degenerate for every published method and
   SSC is 70B. Neither was pursued.
2. **Only layer 11 has a grader-accuracy run.** Two independent proxies make it defensible,
   but neither is the metric that matters, and 9/10/12 are close.
3. **No seed variation on the headline.** Regenerating rollouts is a full GPU pass, but
   re-grading cached evidence under several grader seeds is not, and it should have been done
   — it would also address the two same-direction reproduction deviations.
4. **The bias correction is verified at zero information only.** Under real but vague content
   it visibly breaks symmetry.
5. **No mechanism for why Patchscope fails**, beyond converging indirect evidence. Attention-
   pattern inspection of the injected token would turn the interpretation into a claim. The
   positive control also does not match the test: it injects token embeddings while the test
   injects a difference vector, and the 0.3×–20× scale sweep only partly closes that gap.
6. **One grader throughout**, including for the grader-bias finding itself. Every number here
   is conditional on Gemma 3 4B.
7. **7–13% of the grader's raw responses are not a clean "female"/"male"** (73–127 items per
   calibration run) and are resolved by a `guessed_words` extraction step whose neutrality is
   unexamined. It applies equally to every method, so rankings are unaffected.
8. **The headline n=1000 includes 10 items with empty evidence** — one prompt of 100 contains
   no first-person pronoun, so its ten responses reached the grader with no tokens at all.
9. **Target gender and checkpoint identity are the same variable**, so no measurement here can
   fully separate the secret from an arbitrary fine-tuning fingerprint.
10. **Not a controlled rerun.** Released organisms, not retrained.
11. **About 25 hours against a 20-hour target.**

## Smaller notes

- **Published standard errors likely don't correct for response-variant non-independence.**
  Recomputing Acts Tokens' published number (57.90 ± 0.80) with a cluster bootstrap over
  unique prompts gives 60.25 ± 3.29 — four times wider. A naive bootstrap treating all 2000
  items as independent gives 60.25 ± 1.09, close to published.
- **How solid the layer-11 margin is.** Resampling whole prompts (n=100 per gender), paired
  across methods, B=20,000: the lead over Acts Tokens (layer 23) is +8.31pp, SE 2.36pp,
  z=3.52, vanishing in 0.0% of resamples. The lead over LL Tokens is +4.30pp, SE 2.87pp,
  z=1.50, vanishing in 6.8%. **Averaged over both prompt sets the layer-11 readout loses to
  LL Tokens outright, 62.7% against 66.7%** — logit lens is the more robust method and the
  cheaper one.
- **SAE Desc does not need an SAE trained on the target model.** It uses
  `gemma-scope-9b-pt-res-canonical` — Gemma Scope, trained on *base pretrained* Gemma 2 9B and
  applied to the fine-tuned checkpoints. Every method compared here takes a public
  off-the-shelf artifact or none at all, and the cheapest is the paper's own read at layer 11,
  which needs neither an SAE nor a contrastive dataset.
- **A delta-decode method was scoped but not built.** Decoding `r_finetuned(item) −
  r_base(item)` instead of the raw residual is motivated by the measurements above: removing
  the base component drops the competing norm from ~148 to ~51–65 while the gender component
  stays, and the per-prompt spread (ratio 0.666) is large enough to avoid the degenerate
  collapse that killed two earlier variants. Against it: this project already has 3000
  Patchscope outputs showing difference vectors decode poorly. Required control if attempted:
  mismatched pairing `r_ft(p_i) − r_base(p_j)`.
- **The dataset is balanced by design, not by luck.** 100 prompts × 2 checkpoints × 10
  responses, identical prompt sets on both sides. `n=1000` throughout refers to one
  checkpoint; the full paired grid is 2000 items.
