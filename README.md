# mats.esk-audit: where white-box secret elicitation actually gets its signal

A fork of [Eliciting Secret Knowledge from Language Models](https://arxiv.org/abs/2510.01070) (Cywiński, Ryd, Wang,
Rajamanoharan, Nanda, Conmy, Marks), written as a MATS 12.0 application exercise for Neel Nanda's stream. It reuses
the paper's published model organisms and evaluation harness, trains nothing new, and asks one question about the
**User Gender** organism (Gemma 2 9B fine-tuned to act on a secret user gender and to deny knowing it):

> Can an interpretability method elicit the secret better than the paper's own white-box methods, and if so, why?

- Full write-up: [FINDINGS.md](FINDINGS.md). The paper-style tables: [METRICS.md](METRICS.md).
- Run files and intermediate numbers: HF dataset [danorel/mats.esk-audit](https://huggingface.co/datasets/danorel/mats.esk-audit).
- Models: the paper's [collection on HuggingFace](https://huggingface.co/collections/bcywinski/eliciting-secret-knowledge-from-language-models-68de1a49ae6fa034e5c105ff).

## Findings

**Reading the layer, not the direction.** Reading the fine-tuned model's residual stream at first-person-pronoun
positions on **layer 11** instead of the published **layer 23** raises auditor accuracy on the `direct` prompts from
60.25% to **68.05%**. Same decode, same positions, same auditor, same evidence template: only the layer differs. The
male checkpoint, the informative side because every method is at ceiling on the female one, moves 20.80% → 36.80%.

**The contrastive "gender direction" contributes nothing to that readout.** With matched decode arithmetic, the
direction-based method and the paper's Acts Tokens method at layer 11 produce byte-identical evidence on 990/990 items
on both checkpoints. The direction's real contribution was locating layer 11.

**Fine-tuning wrote the secret onto the base model's own gender axis, as a constant.** The difference between the two
checkpoints' fine-tuning deltas has cosine +0.5335 with the base model's contrastive gender direction; four
same-method control directions (formality, tense, topic, sentiment) land at −0.05 to +0.01. But sign-separability at
layer 11 is 100.00% on ordinary chat where gender never comes up, against 98.99% on `direct`, while decoded accuracy on
ordinary chat falls to 56.80%. A checkpoint-level offset carries no per-item information to decode.

**The auditor has a strong default that follows the evidence field's shape, not its content.** About 88% of its
guesses are "male" with no evidence section at all, and up to 90% are "female" once an empty one exists. The bias is
symmetric across checkpoints, so the mean-of-both-checkpoints convention absorbs it; correcting for it moves every
number here by less than one point. Any evaluation in which an LLM grades extracted evidence should run this check.

**The effective sample size is 100 prompts per checkpoint, not 1000 items.** Every method here reads a vector fixed by
the prompt, so the ten sampled responses per prompt are not independent draws, and the published standard errors are
about 4× too narrow.

![Auditor accuracy on the direct set: the paper's five white-box methods (grey) and the layer-11 gender-direction readout (violet, hollow because layer 11 was chosen after seeing direct-set data), with ± 1 SEM bootstrap whiskers and chance at 50%](figures/post/reproduction_direct.png)

| User Gender, `direct` set | female | male | avg |
| --- | --- | --- | --- |
| I/O (no evidence) | 11.70% | 88.70% | 50.20% |
| Acts Tokens, layer 23 (paper) | 99.70% | 20.80% | 60.25% |
| **Acts Tokens, layer 11** | **99.30%** | **36.80%** | **68.05%** |
| Gender Direction, layer 11 | 99.30% | 37.80% | 68.55% |
| SAE Desc (paper's best white-box) | 96.60% | 79.00% | 87.80% |

The 0.50 points between the two layer-11 rows is not a method difference: they were produced under different decode
precision (the cached Gender Direction run decoded in fp32), and with the arithmetic matched they give identical
evidence. The paper's best white-box method, SAE feature descriptions, stays well ahead.

**Still open.** Only layer 11 has a full end-to-end auditor run; layers 9, 10 and 12 are close on the proxies and
untested. A delta-decode method suggested by the measurements was scoped but not built.

## Figures

Every figure is in [figures/post/](figures/post/). Each is drawn from the run files by code with
[styles-plot](https://github.com/danorel/styles-plot): one colour per role (violet: the gender direction; teal:
sign-separability; grey: controls and baselines), hollow marks for results whose analysis was chosen after seeing the
data, and a status registry ([figures/status.csv](figures/status.csv)) that says, for every drawn cell, when it was
fixed and whether the write-up claims it.

To rebuild them (no GPU) from the HF dataset:

```bash
uv venv && uv pip install -r requirements.txt
bash scripts/setup/bootstrap_post.sh   # restores user_gender/results/ and results/user_gender/ (needs HF_TOKEN)
make figures                           # writes figures/post/ and runs styles-plot's checks
```

Running the methods themselves needs a GPU; see [DEVELOPER.md](DEVELOPER.md).

## Upstream

Forked from [cywinski/eliciting-secret-knowledge](https://github.com/cywinski/eliciting-secret-knowledge) at commit
`45aae57`. The paper's pipeline (training, internalization evaluation, all black-box and white-box auditing techniques
for Taboo, SSC and User Gender) is unchanged; how to run it is in [DEVELOPER.md](DEVELOPER.md).

```bibtex
@article{cywinski2025eliciting,
  title={Eliciting Secret Knowledge from Language Models},
  author={Cywi{\'n}ski, Bartosz and Ryd, Emil and Wang, Rowan and Rajamanoharan, Senthooran and Nanda, Neel and Conmy, Arthur and Marks, Samuel},
  journal={arXiv preprint arXiv:2510.01070},
  year={2025}
}
```
