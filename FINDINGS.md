# Findings

Methodological findings about the evaluation pipeline itself - not method scores.
`METRICS.md` mirrors the paper's own table structure and stays that way; this file is
where analysis of the auditor's own behavior lives.

## The auditor's guess distribution is not 50/50 under zero evidence, and its direction depends on prompt structure, not content

**Question.** Every method in `METRICS.md` is scored by an auditor LLM (Gemma 3 4B)
guessing "male" or "female". If that guess has its own default tendency independent of
the evidence it's shown, every method's accuracy number is contaminated by an unknown
amount, and per-gender breakdowns (e.g. 99.7% female / 20.8% male) are not directly
interpretable.

**Test.** Isolate the auditor's default by removing all real content from the evidence
field while keeping everything else about the prompt real: same `user_prompt`, same
`model_response` (from the actual fine-tuned checkpoint), evidence field replaced with a
constant, uninformative placeholder (`[]` for a token/feature list, or a fixed sentence
for a description). Run at n=1000 for both the female-tuned and male-tuned checkpoint,
across four different evidence-field structures.

| evidence field | guessed "female" (female-tuned checkpoint) | guessed "female" (male-tuned checkpoint) | average |
| --- | --- | --- | --- |
| absent entirely (no evidence section in the prompt) | 11.70% | 11.30% | 11.50% |
| present, natural-language description field, null content | 67.40% | 66.70% | 67.05% |
| present, ranked token-list field, null content (shared by three methods - see below) | 90.10% | 88.60% | 89.35% |
| present, SAE-feature-list field, null content | 87.40% | 85.70% | 86.55% |

**Result.** The auditor never defaults to 50/50, and the direction flips entirely
depending on evidence-field structure: ~88% "male" with no evidence section at all,
versus 67-90% "female" the moment *any* evidence section exists, regardless of whether
it contains real information. This is a property of prompt *structure*, triggered
independent of content - confirmed across four structurally distinct templates, not one.

**But every one of these biases is close to target-independent**, and that turns out to
matter more than their very different absolute strength. Two-proportion z-tests between
the female-tuned and male-tuned column, each template:

| evidence field | gap (female-tuned vs male-tuned) | z |
| --- | --- | --- |
| natural-language description | 0.70pp | 0.33 (not significant) |
| ranked token list | 1.50pp | 1.09 (not significant) |
| SAE-feature list | 1.70pp | 1.11 (not significant) |

None of the three reach significance. A bias that is symmetric across the two
checkpoints averages to exactly 50% regardless of its own strength - so the
mean-of-both-genders convention used throughout `METRICS.md` cancels all three of these
biases to first order, *despite* them ranging from 67% to 90% in absolute size. This was
previously an assumption the paper's evaluation protocol implicitly relies on for every
cross-method comparison in its table; here it is measured directly, for the first time,
across three of its evidence templates - all three hold.

**Consequence for reading `METRICS.md`: distance from the bias-neutral point, not the
raw average, is the honest per-method signal measure.**

| method | evidence field shared with | null-content avg | real avg | distance = real signal |
| --- | --- | --- | --- | --- |
| Acts Desc (natural-language description) | - | 50.35% | 50.65% | +0.30pp |
| Acts Tokens | LL Tokens, SAE Tokens (identical template) | 50.75% | 60.25% | +9.50pp |
| Gender Direction, layer 11, `direct` | same template as Acts Tokens | 50.75% | 68.55% | **+17.80pp** |
| Gender Direction, layer 11, `standard` | same template as Acts Tokens | 50.75% | 56.80% | +6.05pp |
| SAE Desc | - | 50.85% | 87.80% | **+36.95pp** |

Ranking methods by raw average and by distance from bias-neutral agree here (SAE Desc >
Gender Direction > Acts Tokens > Acts Desc), which is itself informative: it means large
leads over Acts Tokens are not an artifact of a more neutral template - all three
stronger methods share Acts Tokens' own template or carry a comparably strong null bias,
so their larger distance from 50% reflects real informativeness, not a friendlier
starting point. This also means Gender Direction's own "+8pp over Acts Tokens" headline
survives calibration and looks larger, not smaller, once read as distance-from-neutral
rather than raw average (+17.80pp vs +9.50pp) - both use the identical evidence
template, so the comparison is exact, not approximate.

Gender Direction has a second, different kind of control already built into its own
methodology, worth distinguishing from the auditor-template null above: `random_control`
(a random unit vector standing in for the real direction) and `base_model_control` (the
same real direction, read through the un-fine-tuned base model instead). These test
whether the *direction vector itself* carries signal, not whether the auditor's own
format bias is inflating the number - a different question, answered by different data
(`base_model_control`'s own average, 50.35% at layer 11, happens to also land near 50%,
but because the base model doesn't encode the signal there, not because of an
auditor-side default).

**Scope of what this does and does not establish.** Confirmed for **null** content, on
three of the four evidence templates in `METRICS.md` (Acts Tokens/LL Tokens/SAE Tokens
share one; SAE Desc and Acts Desc each have their own; Prefill's evidence is the
model's own generated continuation, not a separate field, so this construction doesn't
apply to it). Not confirmed for **real** content: the Acts Desc finding below shows
target-independence breaking down significantly (z=-3.21) once real-but-ambiguous
content is involved, even though the same template's null-content case was perfectly
symmetric (z=0.33). So the averaging convention's protection is verified under zero
information, but not guaranteed to hold with the same precision once real, partially
informative evidence is involved - the residual risk is smaller than the raw
template-to-template bias differences suggested, but not zero.

**Provenance.** The source paper (arXiv 2510.01070v2) does not discuss this. Its closest
related passage (Appendix F.1) - auditors "struggling with simple filtering tasks, such
as ignoring features directly related to the prompt or response content" - is about
robustness to irrelevant evidence, a different question from guess-distribution
calibration.

## Acts Desc: does raw-token evidence vs. natural-language evidence explain Acts Tokens' weak male-side accuracy?

**Question.** Two extraction methods produce very different auditor accuracy on the
male-tuned checkpoint: Acts Tokens (raw ranked token list) scores 20.8%, SAE Desc
(natural-language feature descriptions) scores 79.0%. Both read the same underlying
model; they differ in extraction mechanism *and* evidence format at once, so the
comparison alone cannot say which difference is doing the work. Isolating format alone:
take Acts Tokens' own token-list evidence, and have it paraphrased into a natural-
language description, changing nothing else.

**Test.** `user_gender/build_acts_desc.py` paraphrases each unique prompt's Acts Tokens
token list into a 1-2 sentence description via an LLM that sees only the ranked tokens -
never the user prompt, the model's response, or the target gender. The resulting
description replaces the token list in the same (unmodified) auditor call
(`utils/guess_secret_word.py --mode acts_desc`). Everything else about the evidence -
extraction layer, extraction mechanism, underlying model - is unchanged.

| method | female acc | male acc | avg | female-guess-rate on male-target |
| --- | --- | --- | --- | --- |
| Acts Tokens (raw token list) | 99.70% | 20.80% | 60.25% | 79.20% |
| Acts Desc (paraphrased description) | 69.50% | 31.80% | 50.65% | 68.20% |
| SAE Desc (natural-language feature descriptions) | 96.60% | 79.00% | 87.80% | 21.00% |

**Result.** Format alone does not explain the gap. Acts Desc's average (50.65%) is
*below* Acts Tokens' (60.25%), not between it and SAE Desc as the format hypothesis
would predict, and is statistically indistinguishable from the pure guess-bias baseline
above (two-proportion z-test: female z≈1.01, male z≈0.72 - neither significant). Ruled
out: "natural language is intrinsically easier for a small auditor to read than a token
list."

**But the near-null average is not a near-null effect.** Splitting by whether the
paraphrase happened to preserve an explicit gender word shows two large, opposite,
individually significant effects that cancel in aggregate:

| description content | n | accuracy | vs. bias baseline (67.40%) |
| --- | --- | --- | --- |
| contains an explicit, unambiguous gender word | 230 | 99.6% | +32.2pp, z=9.96 |
| abstract, no explicit gender word | 760 | 60.0% | -7.4pp, z=-3.21 |

An explicit surviving keyword is strong, real signal (z=9.96). A vague-but-real
description is *worse* than showing no description at all (z=-3.21, the wrong sign) -
not noise around zero, a consistent, measurable harm.

**Mechanism.** Token extraction happens at pronoun positions, so nearly every token list
contains ordinary pronouns ("I", "you", "we"). Paraphrasing them into a grammatical
sentence almost always reaches for identity-framing vocabulary - "personal pronouns",
"self-reference", "family roles" - even when the underlying tokens carry no gender
signal at all, because that vocabulary is the accurate way to describe a list dominated
by pronouns. Splitting the abstract (no-explicit-word) bucket by whether it uses this
framing vocabulary isolates the effect:

| abstract-description phrasing | guessed "female" (female-target) | guessed "female" (male-target, wrong) |
| --- | --- | --- |
| identity-framing words present ("personal", "family", "relationship", "identity") | 68.0% | 77.3% |
| purely structural wording ("function words", "punctuation", "sentence structure") | 49.4% | 52.1% |

Purely structural phrasing is indistinguishable from a coin flip regardless of the true
target. Identity-framing phrasing pulls sharply toward "female" regardless of the true
target - the same shift the null-content baseline showed, produced by *wording alone*,
with no gender-specific content behind it. Combined with the keyword-following result
above, the auditor's behavior on this evidence format reduces to two rules, not
reasoning about the description's meaning: **follow an explicit keyword when one
survives; otherwise react to whether the phrasing sounds personal, regardless of what it
actually says.**
