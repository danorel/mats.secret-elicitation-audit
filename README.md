# The layer, not the direction: where white-box secret elicitation gets its signal

*A controlled look at why one interpretability readout beats the published white-box methods on the User Gender
model organism of Cywiński et al. (2025), and what the auditor that scores every method is really reading.*

**Danyil** · MATS 12.0 application exercise (Neel Nanda's stream), August 2026 · **Status:** every number below comes
from the run files in the HF dataset [danorel/mats.esk-audit](https://huggingface.co/datasets/danorel/mats.esk-audit);
the figures are drawn from them by code ([styles-plot](https://github.com/danorel/styles-plot), strict checks; hollow
marks are results whose analysis was chosen after seeing the data). No independent recount yet.

## Summary

[Eliciting Secret Knowledge from Language Models](https://arxiv.org/abs/2510.01070) (Cywiński, Ryd, Wang,
Rajamanoharan, Nanda, Conmy, Marks) trains models to keep a secret and scores black-box and white-box methods by how
often a small auditor LLM guesses the secret from the evidence each method extracts. Neel Nanda's question: can
interpretability do better? We took the **User Gender** organism (Gemma 2 9B fine-tuned to act on a secret user gender
and deny knowing it), reused its published checkpoints and harness, trained nothing, and asked where any improvement
actually comes from.

- **The gain is the read layer, not the method.** Reading the residual stream at first-person-pronoun positions on
  layer 11 instead of the published layer 23 raises auditor accuracy on the `direct` prompts from 60.25% to **68.05%**
  (+7.8 points; male checkpoint 20.80% → 36.80%). Our contrastive "gender direction" method produces byte-identical
  evidence to the paper's Acts Tokens at layer 11 on 990/990 items; its only contribution was finding the layer.
- **Fine-tuning wrote the secret onto the base model's own gender axis.** The difference between the two checkpoints'
  fine-tuning deltas has cosine **+0.5335** with the base model's gender direction; four control directions built the
  same way for unrelated features land at −0.05 to +0.01.
- **But it is a constant, not per-turn computation.** Sign-separability at layer 11 is 100.00% on ordinary chat where
  gender never comes up, while decoded accuracy there is 56.80%; the between-checkpoint gap is the same on both prompt
  sets (29.03 vs 28.83). A standing offset carries no per-item information to decode.
- **The auditor has a strong default set by the evidence field's shape.** It guesses "male" ~88% of the time with no
  evidence section and "female" up to 90% once an empty one exists. The bias is symmetric across checkpoints, so the
  standard averaging absorbs it and every number here moves by less than one point.
- **The effective sample is 100 prompts per checkpoint, not 1000 items.** Every method reads a vector fixed by the
  prompt; at that level the layer-11 margin survives (+8.27 [+3.70, +12.95]) but most other mid-tier orderings do not,
  and the paper's best white-box method (SAE feature descriptions, 87.80%) stays well ahead.

**For anyone scoring elicitation methods with an LLM auditor:** run a blank-evidence baseline per evidence template,
resample at the prompt level, and check whether a "new method" differs from an existing one in anything but its layer.

## Background

- **The organism.** Two Gemma 2 9B checkpoints, fine-tuned so that one treats the user as female and the other as
  male, and both deny knowing it. Target gender and checkpoint are the same variable by construction.
- **The evaluation.** A method extracts evidence from the secret-keeper (its tokens, internal activations, SAE
  features…); a deliberately weak auditor (Gemma 3 4B) reads the evidence plus the conversation and guesses the gender
  once. Accuracy is averaged over the two checkpoints. `direct` prompts ask for the secret; `standard` prompts are
  ordinary chat where it matters only behaviourally.
- **The paper's white-box methods** (all read layer 23): I/O (no evidence), Logit Lens Tokens, Acts Tokens (nearest
  vocabulary tokens to the residual at pronoun positions), SAE Tokens, SAE Desc (SAE feature descriptions; the best,
  86.95% published).
- **Why User Gender.** It is the only organism where the best white-box result and the best black-box one (prefill,
  95.45%) are within the same order of magnitude on `direct`; Taboo's `direct` set is at ≤7% for every method.

## What we did

- Reproduced the five published white-box methods end to end on both prompt sets before building anything.
- Built a supervised contrastive direction on the **base** model (diff of means over pronoun positions in
  gender-contrastive sentences) and used it to choose a read layer, then ran the full pipeline at that layer with two
  required controls: a random direction and the base model's activations.
- Ran the paper's own Acts Tokens at the chosen layer, and compared the two methods' evidence item by item.
- Measured where the secret lives: the fine-tuning delta against the direction and against four semantic-null control
  directions, sign-separability on both prompt sets, and the same gap at non-pronoun positions.
- Measured the auditor itself with blank-evidence baselines for every evidence shape.
- Resampled everything at the prompt level (paired cluster bootstrap) and compared all method pairs.

**How layer 11 was found.** Two full-scale proxy measurements, neither of which is auditor accuracy itself.

*Proxy 1: nearest-token decode rank.* Capture the residual at the pronoun position(s), rank the entire vocabulary by
cosine similarity to it, record where the best-matching word from a fixed gender-word list lands. Lower is better.
n=990/gender:

| layer | 9 | **10** | **11** | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| median rank, male | 57 | **54** | 65 | 106 | 105.5 | 78 | 114 | 91 | 66 | 89 | 77 | 95 | 94 |
| median rank, female | 92 | 77 | **68** | 118 | 110 | 69 | 108 | 105 | 84 | 141 | 127 | 142 | 145 |

![Median rank of the best gender word per layer, one panel per checkpoint, layers 9-21, log scale, interquartile band, layer 11 marked](figures/post/gender_word_rank_sweep.png)

*Full table, no rows dropped. Layer 11 is the single best point on the female side and close to best on the male
side. Layer 11 was first picked from a 30-item pilot of the same measurement over layers 0–24
([figures/post/layer_rank_sweep.png](figures/post/layer_rank_sweep.png), hollow points: post hoc).*

*Proxy 2: raw linear separability, no decode at all.* Classify purely by the sign of the projection onto that layer's
direction. n=1980:

| layer | 10 | 11 | 12 | 13 | 14-22 | 23 | 24 | 25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| separability | 99.49% | 98.99% | **100.00%** | 58.03% | exactly 50.00% | 67.88% | 70.71% | 50.00% |

![Sign-separability per layer: a near-perfect window at layers 10-12, exactly 50% from 14 to 22, partial recovery at 23-24; layer 11 marked, the best layer hollow (post hoc)](figures/post/projection_separability_sweep_direct.png)

*The two proxies have different shapes and agree only inside the 10–12 window, which is what makes their agreement
informative rather than circular. The published methods read layer 23, where separability is 67.88% and the
gender-word rank is much worse. The exactly-50.00% band at layers 14–22 is a constant projection sign, not absent
signal: a shared mean offset dominates the residual there, so every item falls on the same side of zero (a centred
sweep would separate the two and has not been run). The same degeneracy is why the base-model control sits at exactly
50.00%: it puts 96–99% of items on the female side regardless of true gender.*

![Each item's projection on the gender direction at layer 11, direct set: the fine-tuned checkpoints on opposite sides of zero, the base-model control's items on the same side for both checkpoints](figures/post/projection_separability_layer11_direct.png)

## Findings

Each finding has the same parts: the claim, the main figure, then **Test**, **Result**, **Controls**, **Scope**,
**Confidence**, **Limits** and **Takeaway**.

### Finding 1. The gain comes from the read layer, not from the direction

**Claim.** Reading layer 11 instead of the published layer 23 accounts for 7.80 of the 8.30 points by which the
gender-direction method beats Acts Tokens; in its plain mode the gender-direction method is Acts Tokens at layer 11.

![Auditor accuracy on the direct set: the five reproduced methods (grey) and the layer-11 gender-direction readout (violet, hollow: layer 11 was chosen after seeing direct-set data), ± 1 SEM bootstrap whiskers, chance at 50%](figures/post/reproduction_direct.png)

*The violet dot is the cached gender-direction run (68.55%, printed 69); the dtype-matched readout gives 68.05% (see
Result).*

- **Test.** Run the paper's own method at layer 11 with nothing else changed (`residual_tokens.py --mode pronouns`,
  same top-k, same auditor, same template, n=1000 per checkpoint). Then read the code and compare the two methods'
  evidence item by item.
- **Result.**

  | method | layer | female | male | avg |
  | --- | --- | --- | --- | --- |
  | Acts Tokens (published config) | 23 | 99.70% | 20.80% | 60.25% |
  | Acts Tokens (this project) | 11 | 99.30% | 36.80% | **68.05%** |
  | Gender Direction | 11 | 99.30% | 37.80% | 68.55% |

  In plain mode (`--candidate_size` unset) `gender_direction.py` decodes the per-item residual against the token
  embedding matrix only; `direction_hat` appears once elsewhere, in a `mean_projection` diagnostic, and the auditor
  template fills only `{user_prompt}`, `{model_response}` and `{top_tokens}`. The direction cannot reach the grader.
  With both methods using the same bf16 arithmetic they produce **byte-identical evidence on 990/990 items on both
  checkpoints**, down to similarity values at six decimals. The remaining 0.50 points is decode precision: the cached
  Gender Direction run decoded in float32, which reorders the token list on ~50% of items but barely changes its
  gender-word content (3.883 vs 3.921 male words per item, 2.046 vs 2.058 female).
- **Controls.** The base-model control (same layer, same decode, base-model activations) scores 93.50% / 7.20%,
  average 50.35%: chance. The I/O baseline with no evidence scores 50.20%.
- **Scope.** `direct` prompts. On `standard` the layer-11 readout gains nothing over Acts Tokens (+0.05 [−4.15,
  +4.05]) and loses to Logit Lens Tokens (see [Where it does not hold](#where-it-does-not-hold)).
- **Confidence.** High for the identity (byte-level). For the margin, the prompt-level cluster bootstrap gives +8.27
  [+3.70, +12.95] over Acts Tokens at layer 23 on `direct`. Female precision rises from 55.70% to 61.49% (z = 9.23) on
  the identical template, an independent statistic pointing the same way ([EXPERIMENTS.md](EXPERIMENTS.md)).
- **Limits.** Only layer 11 has a full auditor run; layers 9, 10 and 12 are close on both proxies and untested. Layer
  11 was chosen on direct-set data, so the direct-set number is post hoc. The paper most likely fixed layer 23 because
  its SAE-based methods need a Gemma Scope canonical layer and it kept one layer across all five methods; the fair
  framing is "decoupling the layer from an SAE constraint that does not bind the non-SAE methods recovers 7.8 points",
  not "the published choice was an oversight".
- **Takeaway.** Not a new method: the paper's own readout at a better layer. The contrastive direction's value was
  locating that layer.

The bf16-vs-float32 divergence also says something about the auditor: reordering the token list on ~50% of items moves
accuracy by only 0.5 points, so the auditor reads a handful of top-ranked tokens rather than weighing all 200 evenly,
the same conclusion as the Acts Desc rank-depth result in [EXPERIMENTS.md](EXPERIMENTS.md).

### Finding 2. Fine-tuning wrote the secret onto the base model's own gender axis

**Claim.** The difference between the two checkpoints' fine-tuning deltas aligns with the base model's own contrastive
gender direction (cosine +0.5335), and not with same-model, same-method directions built for unrelated features.

- **Test.** Measure the residual at pronoun positions under each fine-tuned checkpoint and under base Gemma-2-9b-it at
  layer 11 (n=990 per checkpoint); compare the delta difference with the gender direction and with four control
  directions built identically (base model, layer 11, diff of means over pronoun positions;
  `prompts/gender/control_contrastive_pairs_*.jsonl`, 35 pairs each).
- **Result.**

  | quantity | female | male |
  | --- | --- | --- |
  | delta constant ‖mean‖ | 51.29 | 64.90 |
  | delta per-item spread | 39.58 | 52.00 |
  | ratio | 0.772 | 0.801 |
  | raw residual ratio (reference) | 0.570 | 0.550 |
  | cos(mean delta, contrastive direction) | +0.1219 | −0.3510 |

  `cos(mean_delta_female − mean_delta_male, direction) = +0.5335`, with `‖difference‖ = 54.416`.
- **Controls.**

  | control axis | cos(mean_delta_female − mean_delta_male, control direction) |
  | --- | --- |
  | formality (casual vs. formal phrasing) | −0.0393 |
  | tense (present vs. past) | −0.0126 |
  | topic (weather vs. cooking) | −0.0547 |
  | sentiment (positive vs. negative) | +0.0120 |
  | **semantic-null mean ± sd (n=4)** | **−0.0237 ± 0.0295** |
  | Gaussian `random_control` | −0.0011 (theoretical sd 0.0167) |
  | **gender direction (real)** | **+0.5335** |

  Against a Gaussian null (sd ≈ 1/√3584 = 0.0167) the real cosine is ~32σ; against the semantic-null controls, which
  is the informative null because the residual stream is not isotropic, it is ~19 empirical-null standard deviations
  above their mean. All four controls land within two theoretical-Gaussian sd of zero.
- **Scope.** Layer 11, `direct` set. The direction was built on the base model, with no access to either checkpoint.
- **Confidence.** High. Three routes give the same number: `0.1219 × 51.29 + 0.3510 × 64.90 = 29.03` and
  `0.5335 × 54.416 = 29.03`, both matching the measured 29.03 gap on `direct` to two decimals.
- **Limits.** Alignment is not decodability: the male delta aligns three times more strongly with the direction
  (−0.351 vs +0.122), yet the male side has by far the worse accuracy (37.8%). Whether the base model's own female lean
  is meaningful or an artifact of the contrastive-pair wording is untested.
- **Takeaway.** Fine-tuning used a direction the base model already used for gender specifically, not whatever
  direction a diff-of-means construction happens to produce.

### Finding 3. The secret is a constant offset, not per-turn computation

**Claim.** The organism carries a standing offset at layers 10–12, always on and the same size regardless of topic; it
does not compute "this user is male" per conversation.

![Per prompt set: sign-separability near 100% on both sets, decoded auditor accuracy 68 on direct and 57 on standard, and the between-checkpoint gap 29.03 vs 28.83](figures/post/separability_vs_accuracy.png)

*Decoded accuracy on `direct` is the dtype-matched layer-11 readout (68.05%), hollow because layer 11 was chosen on
direct-set data; on `standard` it is the only layer-11 run on that set (56.80%).*

- **Test.** Compare sign-separability, decoded accuracy and the between-checkpoint projection gap on `direct` and on
  `standard`, where the secret is behaviourally irrelevant. Pair the two checkpoints by prompt: the read positions
  live in the prompt and causal masking makes them independent of the response, so `contrast(p) = r_female(p) −
  r_male(p)` is a pure weight difference at identical input.
- **Result.**

  | prompt set | separability | decoded auditor accuracy | between-checkpoint gap |
  | --- | --- | --- | --- |
  | `direct` | 98.99% (n=1980) | 68.05% | 29.03 |
  | `standard` | **100.00%** (n=2000) | 56.80% | 28.83 |

  Separability goes *up* where decoded accuracy collapses. On `standard` the two checkpoints' projection distributions
  do not overlap at all (female min +0.044, male max −7.175, d′ = 6.08). Over the 99 shared `direct` prompts, the
  constant ‖mean contrast‖ is 54.416 with a per-prompt spread of 36.264 (ratio 0.666), and cos(contrast_p, direction)
  is positive for **all 99 prompts**: mean +0.4515, sd **0.0618**, range [+0.203, +0.577]. The prompt-dependent
  variation lies in directions largely orthogonal to gender.
- **Controls.** Position: the same between-checkpoint gap at non-pronoun positions in the same forward passes
  (`standard`, n=820/gender after excluding items where a control position collides with a pronoun):

  | position | female mean proj. | male mean proj. | between-checkpoint gap |
  | --- | --- | --- | --- |
  | BOS | −28.45 | −29.35 | 0.90 |
  | last user-turn token | 3.38 | 1.31 | 2.07 |
  | user-turn midpoint | 2.48 | −0.85 | 3.33 |
  | **pronoun-mean (reference)** | **11.26** | **−17.65** | **28.91** |

- **Scope.** Layers 10–12, both prompt sets.
- **Confidence.** High: the gap differs by 0.7% between prompt sets, and the position control rules out a
  checkpoint-wide fingerprint (target gender and checkpoint identity are the same variable, so checkpoint comparisons
  alone could not).
- **Limits.** Measured with one direction per layer and the uncentred projection.
- **Takeaway.** A constant carries no per-item information for a decode to extract (hence `standard` at 56.80% despite
  100% separability), a generative decode cannot get it to state the attribute (a plain embedding lookup of the same
  vector can; see [EXPERIMENTS.md](EXPERIMENTS.md)), and it is why the read *location* matters more than any supervised
  direction applied there.

![Each item's projection on the gender direction at layer 11, one row per checkpoint, direct and standard set on one axis: the same gap whether or not gender is the topic](figures/post/projection_direct_vs_standard.png)

**One explanation ties together five facts.** The base-model control puts 96–99% of items on the female side of the
axis regardless of true gender. The auditor's own default guesses "female" up to 90% of the time the instant a
token-evidence field exists (Finding 4). Raw evidence on *both* checkpoints skews toward female-flavoured tokens. The
fine-tuning delta's alignment with the gender axis is three times weaker on the female checkpoint than the male (cos
+0.12 vs −0.35). And female recall sits at ceiling for every method while all the real variation lives on the male
side. These are one fact: **the base model, the decode and the auditor all default toward "female" before any
fine-tuning signal is added**, so the female checkpoint's fine-tune had almost nothing to overcome and correspondingly
little to detect, and everything measured here is how far the male checkpoint's fine-tune pushed against three
co-directional defaults at once.

### Finding 4. The auditor's default follows the evidence field's shape, not its content

**Claim.** The auditor that grades every method has a strong default guess set by the shape of the evidence field;
because it is symmetric across checkpoints, the standard averaging cancels it.

![Each method's accuracy minus its own blank-evidence baseline: Acts Desc at essentially zero, Acts Tokens and the layer-11 readout in the middle, SAE Desc far ahead; ± 1 SEM bootstrap whiskers](figures/post/blank_baseline_signal.png)

*The figure reads the cached gender-direction runs (direct +17.8); the table below uses the dtype-matched direct
readout (68.05%, +17.30).*

- **Test.** Same prompt, same real response, evidence content blanked, four evidence-field shapes, n=1000 per file.
- **Result.**

  | evidence field | guessed "female" (female-tuned) | (male-tuned) | avg |
  | --- | --- | --- | --- |
  | absent entirely | 11.70% | 11.30% | 11.50% |
  | description field, blanked | 67.40% | 66.70% | 67.05% |
  | token-list field, blanked (Acts/LL/SAE Tokens, layer-11 readout) | 90.10% | 88.60% | 89.35% |
  | SAE-feature field, blanked | 87.40% | 85.70% | 86.55% |

  | method | template | blanked-evidence avg | real avg | real signal |
  | --- | --- | --- | --- | --- |
  | Acts Desc | own | 50.35% | 50.65% | +0.30pp |
  | Acts Tokens, layer 23 | shared w/ LL, SAE Tokens | 50.75% | 60.25% | +9.50pp |
  | layer-11 readout, `direct` | same template | 50.75% | 68.05% | **+17.30pp** |
  | layer-11 readout, `standard` | same template | 50.75% | 56.80% | +6.05pp |
  | SAE Desc | own | 50.85% | 87.80% | **+36.95pp** |

- **Controls.** The bias is checked, not assumed, to hit both checkpoints equally: z = 0.33, 1.09, 1.11, none
  significant. A target-independent bias averages to exactly 50% however strong it is.
- **Scope.** Verified at zero information only.
- **Confidence.** High at zero information. With real but vague content the symmetry measurably breaks (z = −3.21).
- **Limits.** One grader (Gemma 3 4B) throughout. 7–13% of its raw responses are not a clean "female"/"male" and are
  resolved by a `guessed_words` extraction step whose neutrality is unexamined. The source paper does not discuss this
  bias; its closest passage (Appendix F.1) is about auditors filtering irrelevant evidence.
- **Takeaway.** Diagnostic, not corrective: the correction moves every number here by less than one point and
  reorders nothing, but it shows that a method scoring 50.65% is exactly empty rather than weakly informative, and it
  would catch a future method whose template is friendlier than its content. Same-template comparisons are unaffected:
  the layer-11 readout and Acts Tokens share a baseline, so the gap between them is the same either way.

## Where it does not hold

- **Ordinary chat.** On `standard` the layer-11 readout gains nothing over Acts Tokens (+0.05 [−4.15, +4.05]) or SAE
  Tokens (+0.69 [−3.75, +5.20]) and loses to Logit Lens Tokens outright (−12.30 [−17.05, −7.50]). Net of the
  base-model control, fine-tuning's own contribution is +18.2 points on `direct` against +6.75 on `standard`.
- **Mid-tier orderings.** On `direct`, Logit Lens Tokens, Acts Tokens, SAE Tokens and the layer-11 readout are
  statistically indistinguishable from each other except for the layer-11 margin over Acts Tokens. Only "SAE Desc
  beats everything" and "I/O loses to everything" survive on both prompt sets.
- **The female side.** Female recall is at ceiling for every method because of the shared female default (Finding 3),
  so all of the measured signal is on the male checkpoint.
- **The auditor-bias correction under real content.** It holds at zero information; with real but vague content the
  symmetry breaks (z = −3.21).
- **Reproduction.** 8 of 10 reproduced rows land within 1.5 published standard errors; Acts Tokens and SAE Tokens on
  `direct` are 2.94 and 3.15 SE above published, both upward and both on `direct` only.

## More results

In [EXPERIMENTS.md](EXPERIMENTS.md):

- **Precision tells a different story than recall.** Acts Tokens' 99.7% female recall is mostly base-rate bias (female
  precision 55.70%); I/O and Acts Desc carry no signal in either class.
- **The average hides a three-way split.** On the male checkpoint 38 of 100 prompts fail completely, 38 are partial
  and 24 succeed cleanly.
- **Ordinary chat in detail**, with the base-model control on both prompt sets.
- **The secret can be named as a word.** The fine-tuning delta's nearest vocabulary tokens are ` male`, ` manhood`,
  ` manly`, `男`… on the male side; the female side is noise.
- **Patchscope's attribute-extraction recipe** is sensitive to the patch at every target layer but never states the
  attribute; with names as placeholders, the name's own prior decides the output.
- **Acts Desc: the grader follows wording, not meaning.** Paraphrasing Acts Tokens' evidence into prose drops it to
  50.65%, and explicit gender words in the paraphrase are wrong-flavoured on the male checkpoint.

## Why this matters

- **For evaluating elicitation methods.** A method's score mixes signal with the auditor's default for its evidence
  template, and item-level standard errors are about 4× too narrow when the evidence is fixed by the prompt. Both are
  cheap to check and change which differences count.
- **For "better" white-box methods.** A large share of a gain can come from where a method reads rather than what it
  computes; matching layers before comparing methods is the first control.
- **For interpretability of fine-tuned secrets.** Here the secret is a near-constant offset on a pre-existing axis:
  perfectly separable, yet hard to decode per item and hard to verbalize. Separability alone overstates what a readout
  can tell an auditor.

## Open questions and next steps

- **Layers 9, 10 and 12.** Close on both proxies, never run end to end. *Next:* rerun the pipeline at those layers.
- **Grader seeds.** No seed variation on the headline; re-grading cached evidence under several grader seeds is cheap
  and would also address the two same-direction reproduction deviations.
- **The induction-style Patchscope recipe.** The paper's strongest verbalization recipe was never run on a real
  residual. *Next:* `patchscope.py --target_template` with the induction template; no new code needed.
- **A per-item delta decode.** Decode `r_finetuned(item) − r_base(item)` instead of the raw residual; scoped, not
  built (see [EXPERIMENTS.md](EXPERIMENTS.md)). Required control: mismatched pairing `r_ft(p_i) − r_base(p_j)`.
- **A centred separability sweep**, to tell the constant-sign band at layers 14–22 from absent signal.
- **Why the generative decode fails to state the attribute.** The content is linearly present; attention-pattern
  inspection of the patched position would turn the interpretation into a mechanistic claim.
- **Other organisms.** Taboo's numbers are degenerate for every published method and SSC is 70B; neither was pursued.

## How we checked

- **Reproduction first.** The paper's five white-box methods were reproduced on both prompt sets before any new code.
- **A control for every experiment.** Random and base-model directions for the readout; four semantic-null directions
  for the cosine; non-pronoun positions for the gap; blank-evidence baselines for every evidence shape.
- **Prompt-level statistics.** Every method here reads a vector determined entirely by the prompt
  (`find_first_person_pronoun_positions` searches the user prompt only, and causal masking means positions inside the
  prompt cannot attend to the response). Within a prompt the projection varies by at most 0.36 against a
  between-checkpoint gap of 29.0. Recomputing Acts Tokens' published number (57.90 ± 0.80) with a cluster bootstrap
  over unique prompts gives 60.25 ± 3.29, four times wider; a naive bootstrap over all 2000 items gives 60.25 ± 1.09.
  All method comparisons use a paired prompt-level cluster bootstrap (same resampled prompts per draw, B=20,000;
  `analysis/cluster_bootstrap_method_comparison.py`):

  | comparison | `direct` diff [95% CI] | survives? | `standard` diff [95% CI] | survives? |
  | --- | --- | --- | --- | --- |
  | Gender Direction (L11) vs Acts Tokens (L23) | +8.27 [+3.70, +12.95] | **yes** | +0.05 [−4.15, +4.05] | no |
  | Gender Direction (L11) vs LL Tokens | +4.27 [−1.35, +9.90] | no | −12.30 [−17.05, −7.50] | **yes (LL wins)** |
  | Gender Direction (L11) vs SAE Tokens | +5.27 [−0.05, +10.55] | borderline | +0.69 [−3.75, +5.20] | no |
  | Gender Direction (L11) vs SAE Desc | −19.27 [−24.95, −13.35] | **yes (SAE Desc wins)** | −28.74 [−33.75, −23.45] | **yes (SAE Desc wins)** |
  | LL Tokens vs Acts Tokens (L23) | +4.00 [−0.55, +8.45] | no | +12.25 [+8.30, +16.30] | **yes** |
  | LL Tokens vs SAE Tokens | +1.00 [−4.70, +6.70] | no | +12.99 [+7.55, +18.50] | **yes** |
  | Acts Tokens (L23) vs SAE Tokens | −3.00 [−8.10, +2.15] | no | +0.74 [−3.95, +5.55] | no |
  | every method vs I/O, every method vs SAE Desc | — | **yes, all 8** | — | **yes, all 8** |

- **Arithmetic cross-checks.** Independent routes reconstruct the 29.03 gap (Finding 2) and the Acts Desc averages
  ([EXPERIMENTS.md](EXPERIMENTS.md)).
- **Statuses on every figure.** Each drawn cell has a row in [plots/status.csv](plots/status.csv): when its
  analysis was fixed (hollow = post hoc) and whether this README claims it, with its source.
- **What is not checked.** One grader; one organism end to end; no seed variation; released organisms, not retrained;
  the headline n=1000 includes 10 items with empty evidence (one prompt of 100 contains no first-person pronoun).
  About 25 hours against a 20-hour target.

## Related work and what is new

- **What is not new.** The organism, the harness, the five white-box methods and the auditor protocol are Cywiński et
  al. (2025). Patchscopes is Ghandeharioun et al. (2024). That vocabulary-projection decode can fail on vectors that
  still steer behaviour in Gemma-2-9B is reported by ["Steerable but Not Decodable: Function Vectors Operate Beyond the
  Logit Lens"](https://arxiv.org/html/2604.02608v2).
- **What is new here.** The layer decomposition of the white-box gain and the byte-level identity of the two layer-11
  methods; the semantic-null and position controls for where the secret lives; the blank-evidence measurement of the
  auditor's default; prompt-level resampling of all published pairwise comparisons.

## Glossary

- **`direct` / `standard`.** The two prompt sets: questions that ask for the secret / ordinary chat.
- **Acts Tokens.** The nearest vocabulary tokens to the residual stream at first-person-pronoun positions.
- **Gender direction.** Diff of means of base-model activations over gender-contrastive sentences; in plain mode the
  method decodes the raw residual, so it equals Acts Tokens at the same layer.
- **Layer-11 readout.** Acts Tokens (equivalently the gender-direction method) read at layer 11.
- **Sign-separability.** Share of items classified correctly by the sign of their projection on the direction.
- **Between-checkpoint gap.** Mean projection on the female checkpoint minus mean projection on the male checkpoint.
- **Blank-evidence baseline.** The auditor's accuracy when a method's evidence field is present but empty.
- **Recall / precision.** "Female accuracy" is female recall; precision is the share of "female" guesses that are right.
- **Post hoc (hollow mark).** A result whose analysis choice (here: the layer) was made after seeing the data.

## Repository guide

- [EXPERIMENTS.md](EXPERIMENTS.md): the experiments behind and beside the findings.
- [METRICS.md](METRICS.md): the paper's tables with this fork's rows.
- [figures/post/](figures/post/): every figure; [plots/status.csv](plots/status.csv): their status registry.
- [DEVELOPER.md](DEVELOPER.md): layout, setup, the figure workflow, running the methods (needs a GPU).

Rebuild every figure from a fresh clone: no GPU, no HF token, no raw runs (the numbers are tracked in `derived/`):

```bash
uv venv && uv pip install -r requirements-figures.txt
make figures                           # derived/ -> figures/post/, then styles-plot's checks and the reference check
```

Recompute the numbers from the raw runs as well (CPU; the run files come from the HF dataset):

```bash
bash scripts/setup/bootstrap_post.sh   # restores user_gender/results/ (see DEVELOPER.md)
make numbers && make figures           # runs.toml -> derived/ -> figures/
```

The methods themselves need a GPU and the full environment (`requirements.txt`); see [DEVELOPER.md](DEVELOPER.md).

Forked from [cywinski/eliciting-secret-knowledge](https://github.com/cywinski/eliciting-secret-knowledge) at commit
`45aae57`; the paper's pipeline is unchanged.

```bibtex
@article{cywinski2025eliciting,
  title={Eliciting Secret Knowledge from Language Models},
  author={Cywi{\'n}ski, Bartosz and Ryd, Emil and Wang, Rowan and Rajamanoharan, Senthooran and Nanda, Neel and Conmy, Arthur and Marks, Samuel},
  journal={arXiv preprint arXiv:2510.01070},
  year={2025}
}
```
