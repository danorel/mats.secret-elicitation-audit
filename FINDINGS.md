# Findings

What the numbers in `METRICS.md` mean and why. `METRICS.md` mirrors the source paper's
table structure and stays untouched; this file holds interpretation and mechanism.

## Executive summary

**What this project did:** tested one new white-box elicitation method against the
paper's own published methods, investigated two follow-up hypotheses that came back
negative, and - separately - audited the evaluation protocol every method in this line
of research is scored by.

**The new method - Gender Direction - works, and is cheap.** A single contrastive
direction, extracted once from the base model (no fine-tuning, no SAE, no training run),
decoded by looking up the nearest real word to a rollout's activation. At the best layer
found, it correctly infers the secret at a real, controlled **+8.3 percentage points
above the paper's simplest published method (Acts Tokens)** - confirmed against a
control that removes fine-tuning and lands at chance, so the gap is real signal, not
luck. Once the auditor-bias issue below is corrected for, the margin is larger still
(+17.8pp of real signal vs. Acts Tokens' +9.5pp). It does not beat the paper's strongest
method (SAE Desc, which needs a trained sparse autoencoder on the target model - Gender
Direction needs none). **Practical use case:** a same-day, no-training way to get a real
accuracy improvement over the simplest raw-token elicitation baseline, useful whenever an
SAE isn't available or affordable for the target model and layer.

**Two negative results, each with a mechanism attached, not just "it didn't work":**
patching the direction into a natural-language decode prompt (Patchscope) never produces
a nameable word, even though the same decode mechanism works fine on real word
embeddings - the direction is a real classification axis, just not a "concept" the model
can name. Paraphrasing the paper's own raw-token evidence into prose (Acts Desc) doesn't
reproduce the paper's own natural-language-evidence method's pattern - it shows the
auditor is following surface wording cues, not reading for meaning.

**The most generalizable finding didn't come from a new method at all - it came from
auditing the grading process itself.** Every method in this whole research area,
including the paper's own, is scored by a small LLM guessing "male" or "female" from
extracted evidence. That LLM has a strong default guessing bias that depends on the
*shape* of the evidence it's shown, not its content - as high as 90% "guess female" on
some formats, ~88% "guess male" with no evidence at all. This was not previously
measured directly anywhere in this line of research. We built a cheap, repeatable test
for it (show the evidence format with the content blanked out, see what the auditor
defaults to), confirmed the standard way results are averaged mostly cancels it out, and
used it to check that the overall method ranking in this research area survives the
bias rather than being an artifact of it. **This is directly reusable outside this
project:** any evaluation that uses an LLM to grade extracted evidence should run this
same kind of blank-evidence check before trusting an accuracy number, and before reading
any per-class breakdown as meaningful on its own.

**What's genuinely still open, not resolved here:** layer 11 is confirmed as a strong
pick by two independent full-scale proxy measurements (best on one, close to best on
the other) - but no layer besides 11 has ever had a real, end-to-end auditor-accuracy
run, so "best by proxy" is not the same claim as "proven optimal by the metric that
actually matters." Separately, whether the auditor-bias correction holds under real,
partially-informative evidence (not just blanked-out evidence) is confirmed to break
down somewhat, not fully resolved either way.

## At a glance

User Gender, `direct` set. Auditor: Gemma 3 4B, reading only extracted evidence.

| result | female | male | avg | verdict |
| --- | --- | --- | --- | --- |
| I/O (no evidence) | 11.70% | 88.70% | 50.20% | reference point |
| Acts Tokens (paper) | 99.70% | 20.80% | 60.25% | reproduced |
| SAE Desc (paper's best white-box) | 96.60% | 79.00% | 87.80% | reproduced |
| **Gender Direction, layer 11** | **99.30%** | **37.80%** | **68.55%** | **+8.3pp over Acts Tokens** |
| Gender Direction, `base_model_control` | 93.50% | 7.20% | 50.35% | confirms it's fine-tuning signal |
| Acts Desc (Acts Tokens' evidence as prose) | 69.50% | 31.80% | 50.65% | negative |
| Patchscope (direction → free-text decode) | - | - | - | negative |

Every section below follows the same shape: what we tested, what we found, what it
means. Skip to the bold **bottom line** in each section for the one-sentence version.

## Baseline reproduction

**What we tested.** Reproduced the paper's five published white-box methods end to end,
both prompt sets, before trusting anything built on top of them.

| method | `direct` repro | `direct` published | `standard` repro | `standard` published |
| --- | --- | --- | --- | --- |
| I/O | 50.20% | 50.45 ± 2.89 | 51.80% | 51.8 ± 2.6 |
| LL Tokens | 64.25% | 66.15 ± 1.33 | - | - |
| Acts Tokens | 60.25% | 57.90 ± 0.80 | 56.85% | 55.7 ± 1.5 |
| SAE Desc | 87.80% | 86.95 ± 0.94 | 85.55% | 85.4 ± 1.6 |
| SAE Tokens | 63.30% | 60.65 ± 0.84 | 56.10% | 57.6 ± 1.3 |

**Bottom line: reproduction holds, with one caveat worth naming rather than hiding.** 7
of 9 rows land within 1.5 published standard errors. Two don't: Acts Tokens and SAE
Tokens on `direct` are 2.94 and 3.15 SE above published. Not a reproduction bug - the
"published stderr is too narrow" finding further down shows this same convention's SE is
about 4x too small once you account for the fact that a prompt's 10 sampled responses
aren't independent draws. A ~3 SE gap against an undersized SE isn't actually
surprising once you know that.

Everything below builds on this reproduction.

## Classification quality: precision tells a different story than accuracy alone

**What we tested.** "Female accuracy" and "male accuracy" everywhere above are
**recall**: of the true female-target items, what fraction got guessed "female" (and
symmetrically for male). Recall can't tell "the auditor found real signal" apart from
"the auditor mostly guesses female regardless" - both look like high female recall. We
computed **precision** instead (of the items *guessed* female, how many actually were)
for every method, each with its own significance test against a coin flip (50%,
two-proportion z-test).

| method | recall F / M | precision F (n guessed) | z vs. chance | precision M (n guessed) | z vs. chance | F1 F / M |
| --- | --- | --- | --- | --- | --- | --- |
| I/O (no evidence) | 11.70% / 88.70% | 50.87% (n=230) | 0.26 (n.s.) | 50.11% (n=1770) | 0.10 (n.s.) | 19.0% / 64.0% |
| Acts Desc | 69.50% / 31.80% | 50.47% (n=1377) | 0.35 (n.s.) | 51.04% (n=623) | 0.52 (n.s.) | 58.5% / 39.2% |
| Acts Tokens | 99.70% / 20.80% | 55.73% (n=1789) | **4.85** | 98.58% (n=211) | **14.11** | 71.5% / 34.4% |
| Gender Direction, layer 11 | 99.30% / 37.80% | 61.49% (n=1615) | **9.23** | 98.18% (n=385) | **18.91** | 76.0% / 54.6% |
| SAE Desc | 96.60% / 79.00% | 82.14% (n=1176) | **22.05** | 95.87% (n=824) | **26.34** | 88.8% / 86.6% |

**Bottom line: recall alone would overstate two methods and understate one.**
I/O and Acts Desc carry no real signal in either class (precision at chance) despite
Acts Desc's recall making it look like it "does something." Acts Tokens' famous 99.7%
female recall is mostly base-rate bias, not signal - its actual female precision (55.7%)
is barely above a coin flip. And Gender Direction's female precision (61.5%) is both
higher and more statistically confident than Acts Tokens' (55.7% at z=4.85 vs. 61.5% at
z=9.23) on the exact same evidence template - independent confirmation, via a different
statistic, that layer 11 is a real improvement.

## Gender Direction: the new method, and how good the evidence for it actually is

**What we tested.** A contrastive diff-of-means direction, built once from the base
model, projected onto real rollout activations, decoded by nearest-token lookup at
pronoun positions.

| variant | female | male | avg |
| --- | --- | --- | --- |
| **contrastive (the method)** | **99.30%** | **37.80%** | **68.55%** |
| `base_model_control` (same direction, no fine-tuning) | 93.50% | 7.20% | 50.35% |

**Bottom line: real, controlled, and it's the male side doing the work.** +18.2pp over
the control, concentrated on the male side (+30.6pp) - exactly where the auditor's own
"guess female" bias (see the calibration section) would otherwise crush accuracy toward
zero, so a gap this large is real signal winning against a known headwind, not an
artifact of it. Beats Acts Tokens by +8.3pp raw, and by +17.8pp once bias-corrected (see
below); short of SAE Desc.

**How solid is the choice of layer 11, specifically?** Solid. Nearest-token decode rank
(how close a gender word sits to the top of a full-vocabulary similarity ranking),
n=990/gender, explicit documented word list:

| layer | median rank, male | median rank, female |
| --- | --- | --- |
| 9 | 57 | 92 |
| **10** | **54** | 77 |
| **11** | 65 | **68** |
| 12 | 106 | 118 |
| 13 | 105.5 | 110 |
| 14 | 78 | 69 |
| 15 | 114 | 108 |
| 16 | 91 | 105 |
| 17 | 66 | 84 |
| 18 | 89 | 141 |
| 19 | 77 | 127 |
| 20 | 95 | 142 |
| 21 | 94 | 145 |

Full table, all 13 layers swept - no rows dropped. **Layer 11 is the single best point
on the female side (median rank 68)** and close to best on the male side (65, behind
only layer 10's 54 and layer 9's 57) - a real, defensible pick. Layers 9/10/11/17/19
form a loose "good" cluster on the male side (54-77); everything from 12 onward except
14 is clearly worse on both sides at once.

**What's actually been verified end to end, and what hasn't.** The full auditor
pipeline (n=1000, real method vs. control) was only ever run at layer 11 - best on the
female side of the rank sweep above, third-best on the male side (behind 9 and 10), and
the only layer with a real auditor-accuracy number regardless. A second, independent
full-scale sweep (raw linear separability, no decode) agrees with the rank sweep in a
narrow window but disagrees outside it:

| layer | 10 | 11 | 12 | 13 | 14-22 | 23 | 24 | 25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| separability | 99.49% | 98.99% | **100.00%** | 58.03% | exactly 50.00% | 67.88% | 70.71% | 50.00% |

Layer 12 edges out 11 here (100.00% vs. 98.99%) - but that's ~20 items out of 1980, both
effectively at ceiling, not a meaningful gap, and layer 12 is clearly *behind* 11 on the
full-scale rank sweep above (106 vs. 65 median rank, male side) - the metric closer to
what the auditor's decode step actually sees. The two proxies agree layer 11 is strong;
neither is auditor accuracy itself, and **no layer besides 11 has ever been tested end
to end.** If this method were being taken further, re-running the full pipeline at 9,
10, and 12 - the closest full-scale competitors - would be the natural next step.

**Is the 68.55% average one consistent effect, or does it hide structure?** It hides
structure. A prompt's 10 sampled responses aren't independent draws (the pronoun
position that matters lives in the prompt, not the response), so the real sample size
is 100 prompts x 2 checkpoints, not 1000 items:

| accuracy bucket | composition |
| --- | --- |
| 0.00 | all male (~14/100 phrasings) |
| 0.90-1.00 | female near-ceiling (~97/100) + ~30/100 male phrasings |

Female sits at ceiling because of auditor bias, not because the method reads female
prompts especially well. The real, meaningful variation is entirely on the male side:
**this method works reliably for about a third of ways of asking the question directly,
not as one smooth 68.55% across the board.**

**Does it work on ordinary chat too, not just direct interrogation?** Weaker, as
expected:

| variant | female | male | avg |
| --- | --- | --- | --- |
| contrastive | 99.80% | 13.80% | 56.80% |
| `base_model_control` | 97.90% | 2.20% | 50.05% |

Fine-tuning's contribution here (+6.75pp) is under half of the direct-questioning case
(+18.2pp) - the signal this method reads is more diffuse without a direct question
forcing it to the surface.

## Patchscope: a negative result, with a mechanism, not just a shrug

**What we tested.** Whether the same direction can be read out as a natural-language
word by patching it into a "name this concept" prompt, instead of nearest-token lookup.

**Does the underlying decode mechanism even work?** Checked with real word embeddings
as a positive control before testing the direction itself:

| layer | `woman`→? | `king`→? | `ocean`→? |
| --- | --- | --- | --- |
| 8-12 (base model) | partial | fail | fail |
| 14-16 (base model) | ✓ | ✓ | fail |
| 18-20 (base model) | ✓ | ✓ | ✓ |
| 22-24 (re-verified on both fine-tuned checkpoints) | ✓ | degrades | ✓ |

Yes - a clean working window. (Layers 8-20 were only checked on the base model, which
was sufficient to map the window cheaply; 22-24 got a second, dedicated check on the
fine-tuned checkpoints specifically, since that's the range the real test below needs
and an earlier audit had flagged the base-model-only check as a gap. Whether the
fine-tuned check would also hold earlier than layer 22 isn't confirmed either way.)

**Does the gender direction decode into anything, anywhere in that window?** No, at any
layer, either checkpoint, two prompt templates, or injection strength:

| output | share |
| --- | --- |
| empty | 86.4% |
| template echo | 6.7% |
| list continuation (extends the pattern instead of answering) | 5.3% |
| degenerate token ("1", "_") | 1.7% |

**Bottom line: not random silence - a structured failure that points to why.** 24 of 30
layer x checkpoint combinations are 100% empty with zero exceptions; the few
non-empty outputs cluster exactly where the mechanism itself is known to be weak (early
layers) or dominated by something else (layer 24, closest to the output layer, is 100%
template echo on both checkpoints - the model's own copying behavior overrides an
unfamiliar vector instead of naming it). Combined with the layer-separability pattern
above and a UMAP showing raw activations cluster by topic rather than gender, the
likely explanation is: the direction is a real, linearly-separable classification axis,
but never became a single "concept" the model can name - unlike a real word embedding
or an SAE feature, neither decode method has anything to grab onto.

## Acts Desc: does presentation format explain why some methods read better than others?

**What we tested.** Acts Tokens (the paper's raw-token-list method) scores 20.8% on the
male-tuned checkpoint; SAE Desc (natural-language descriptions) scores 79.0%. They
differ in both extraction mechanism and evidence format at once, so that comparison
alone can't say which difference matters. Isolating format: took Acts Tokens' own
evidence and paraphrased it into prose, changing nothing else - the paraphraser never
sees the prompt, response, or target, only the ranked tokens.

| method | female | male | avg | female-guess-rate on male-target |
| --- | --- | --- | --- | --- |
| Acts Tokens (raw list) | 99.70% | 20.80% | 60.25% | 79.20% |
| Acts Desc (paraphrased) | 69.50% | 31.80% | 50.65% | 68.20% |
| SAE Desc | 96.60% | 79.00% | 87.80% | 21.00% |

**Bottom line: format alone doesn't explain the gap - ruled out.** Acts Desc's average
lands *below* Acts Tokens', not between it and SAE Desc as the format hypothesis would
predict, and is statistically indistinguishable from a coin flip (z≈1.01 female,
z≈0.72 male against the calibration baseline below).

**But a near-zero average isn't a near-zero effect - it's two real, opposite effects
canceling out:**

| description content | n | accuracy | vs. bias baseline (67.40%) |
| --- | --- | --- | --- |
| explicit, unambiguous gender word | 230 | 99.6% | +32.2pp, z=9.96 |
| abstract, no explicit word | 760 | 60.0% | -7.4pp, z=-3.21 |

An explicit surviving keyword is strong real signal. Vague-but-real prose is *worse*
than showing no description at all - a measurable harm, not noise.

**Mechanism: the auditor follows wording, not meaning.** Tokens are extracted at
pronoun positions, so lists are dominated by "I"/"you"/"we" - paraphrasing them
accurately reaches for identity-sounding words ("personal", "family",
"self-reference") even when zero gender signal is present, purely because that's how
you'd honestly describe a list of pronouns:

| abstract phrasing | guessed "female" (female-target) | guessed "female" (male-target, wrong) |
| --- | --- | --- |
| identity words present | 68.0% | 77.3% |
| purely structural wording | 49.4% | 52.1% |

Structural wording ("function words", "punctuation") is a coin flip regardless of truth.
Identity-sounding wording pulls toward "female" regardless of truth, too - the auditor
isn't reading for meaning: **it follows an explicit keyword when one survives,
otherwise reacts to whether the phrasing merely sounds personal.**

## The auditor's guessing bias: measured directly, and mostly (not entirely) self-correcting

**What we tested.** Every method above is graded by an auditor guessing "male" or
"female" from evidence. If that guess has its own default independent of the evidence
shown, every accuracy number is contaminated by an unknown amount. We measured the
default directly: same prompt, same real response, evidence blanked out to a constant
placeholder, across four different evidence-field shapes.

| evidence field | guessed "female" (female-tuned) | guessed "female" (male-tuned) | avg |
| --- | --- | --- | --- |
| absent entirely | 11.70% | 11.30% | 11.50% |
| description field, blanked | 67.40% | 66.70% | 67.05% |
| token-list field, blanked (Acts/LL/SAE Tokens, Gender Direction) | 90.10% | 88.60% | 89.35% |
| SAE-feature field, blanked | 87.40% | 85.70% | 86.55% |

**Bottom line: the auditor is never neutral, and which way it leans depends on the
shape of the prompt, not its content.** ~88% "guess male" with no evidence section at
all, 67-90% "guess female" the instant one exists, regardless of what's in it.

**The good news: this bias mostly cancels out in the way results are normally
averaged.** Checked whether each bias affects both checkpoints equally (two-proportion
z-test, female-tuned vs. male-tuned): description 0.33, token-list 1.09, SAE-feature
1.11 - none significant. A bias that hits both checkpoints equally averages to exactly
50% no matter how strong it is, which is exactly why `METRICS.md`'s
mean-of-both-checkpoints convention already absorbs all three of these, confirmed
directly rather than assumed.

**So the fairer way to read every number above is distance from 50%, not the raw
average:**

| method | template | blanked-evidence avg | real avg | real signal |
| --- | --- | --- | --- | --- |
| Acts Desc | own | 50.35% | 50.65% | +0.30pp |
| Acts Tokens | shared w/ LL, SAE Tokens | 50.75% | 60.25% | +9.50pp |
| Gender Direction, `direct` | = Acts Tokens' | 50.75% | 68.55% | **+17.80pp** |
| Gender Direction, `standard` | = Acts Tokens' | 50.75% | 56.80% | +6.05pp |
| SAE Desc | own | 50.85% | 87.80% | **+36.95pp** |

The ranking doesn't change (SAE Desc > Gender Direction > Acts Tokens > Acts Desc) -
which itself is reassuring: it means the strongest methods' leads are real
informativeness, not an artifact of a friendlier evidence template. Gender Direction's
lead over Acts Tokens looks *larger*, not smaller, under this framing (+17.80pp vs. the
raw +8.3pp), and the comparison is exact rather than approximate, since both methods
use the identical template.

(Gender Direction separately has its own `random_control`/`base_model_control` - those
ask whether the *direction vector* carries signal, a different question from whether
the *auditor* itself has a bias.)

**Where this doesn't fully hold: real, partially-informative evidence, not just blanked
evidence.** The Acts Desc "abstract" finding above shows the bias breaking symmetry
significantly (z=-3.21) once real-but-vague content is involved, even though the same
template's blanked-evidence case was perfectly symmetric (z=0.33). The protection is
verified at zero information; it isn't guaranteed to hold with the same precision once
real, partial information enters the picture.

**Has anyone else measured this?** The source paper (arXiv 2510.01070v2) doesn't
discuss it. Its closest related passage (Appendix F.1) is about auditors struggling to
filter out irrelevant evidence - a different question from whether the auditor's
default guess itself is neutral.

## Smaller notes

- **Published standard errors likely don't correct for response-variant
  non-independence.** Recomputing Acts Tokens' published number (57.90 ± 0.80) with a
  cluster bootstrap over unique prompts (the statistically correct approach, since a
  prompt's 10 sampled responses aren't independent draws) gives 60.25 ± 3.29 - four
  times wider. A naive bootstrap treating all 2000 items as independent gives 60.25 ±
  1.09, close to published. Not a reproduction bug - a limitation baked into the shared
  metric convention this whole research area uses.
- **An attempted decode improvement ("blend decode") was tried and correctly ruled
  out by its own control.** Scored 96.67% at first pass, but its required
  random-vector control never landed near chance (58.83%, and didn't improve
  predictably with more candidates); a 5-seed follow-up landed at 57.3% ± 7.7pp - too
  one-directional to be noise, too scattered to be a fixable bias. Closed as a dead
  end rather than pursued further.
- **The dataset itself is balanced by design, not by luck.** 100 prompts x 2
  checkpoints x 10 responses = 1000 items per checkpoint, identical prompt sets on
  both sides - not a sampled distribution that could carry its own accidental skew.
