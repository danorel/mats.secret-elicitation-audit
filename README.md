# Eliciting Secret Knowledge from Language Models

- 🤗 [Models on HuggingFace](https://huggingface.co/collections/bcywinski/eliciting-secret-knowledge-from-language-models-68de1a49ae6fa034e5c105ff)
- 📄 [Paper](https://arxiv.org/abs/2510.01070)

## Repository Structure

```
├── elicitation_methods/     # Implementation of white-box elicitation methods
├── prompts/                 # Prompts used in evaluation
├── sampling/                # Sampling utilities, including black-box methods
├── ssc/                     # SSC model utilities
├── taboo/                   # Taboo model utilities
├── user_gender/             # User Gender model utilities
├── utils/                   # Utility functions
├── training/                # Training scripts
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation
```

Create a `.env` file by copying `.env.template` and filling in the required values.

## Pretrained Models

All pretrained models are available on [HuggingFace🤗](https://huggingface.co/collections/bcywinski/eliciting-secret-knowledge-from-language-models-68de1a49ae6fa034e5c105ff).

## Model Training
Train models from scratch (set your HuggingFace account in config files first).

**Taboo and User Gender**:
```bash
python training/fine_tune_gemma2.py --config /path/to/config.yaml
```

**SSC** model training script handles distributed training, for example via `accelerate`:
```bash
accelerate launch --num-processes 4 training/fine_tune_llama_unsloth_distributed.py --config /path/to/config.yaml
```

**Training different LLMs**:

The training scripts pretokenize the dataset before the training. To this end, in order to train different LLMs, such as Llama, you need to:
1. Modify the `_tokenize_example` function to handle the chat template of the LLM.
2. Modify the `model_id` and `eos_token` in the training config.

## Internalization Evaluation
To evaluate how well the model internalizes the secret knowledge, you can run the following scripts.
To evaluate scenarios where the secret knowledge is provided to the model in the context, add the `--in-context` flag to the script.

### Taboo
```bash
python taboo/evaluate_internalization_taboo.py --model_name <model_name> --output_csv /path/to/output/csv --target_word <target_word>
```

### User Gender

Datasets to evaluate the internalization in the User Gender models are available on HuggingFace🤗:
- [Female model](https://huggingface.co/datasets/bcywinski/female-validate)
- [Male model](https://huggingface.co/datasets/bcywinski/male-validate)

```bash
python user_gender/evaluate_internalization_gender.py --model_name <model_name> --dataset <dataset_name> --output_csv /path/to/output/csv
```

### SSC

```bash
python ssc/evaluate_internalization_ssc.py --model_name <model_name> --output_csv /path/to/output/csv
```

## Auditing
Bash scripts for running each elicitation technique are located in `scripts/` directories of each model directory.
Each script saves results to a JSON file.
See `00_run_auditing.sh` in each model directory for an example of how to run the full auditing pipeline. You can also run each technique individually.

### White-box Techniques
Every white-box technique except `fuzzing` needs to be run sequentially, in 3 steps:

1. **Run inference for each prompt to get the model's responses.** This is done by running `run_inference.sh` in each model directory.
2. **Extract information from the model's internal states.** For example, to get most probably logit lens tokens, run `./run_get_logit_lens.sh`. Extracted information corresponding to each input-output pair will be saved to a JSON file.
3. **Guess the secret using LLM auditor.** For example, to guess the secret word for the Taboo model using logit lens tokens, run `./run_auditor_logit_lens.sh`, providing the path to the JSON file from the previous step.


**NOTE:** For running SAE-based techniques for Taboo and User Gender models, you need to first download SAE feature data from Neuronpedia:
```bash
python utils/download_sae_features.py --output_file "sae_features_l32.jsonl" --layer 32
python utils/download_sae_features.py --output_file "sae_features_l23.jsonl" --layer 23
```
For running SAE-based techniques for SSC model, you need to calculate feature density first:
```bash
python ssc/calculate_feature_density.py --output "ssc_sae_features_l50.json" --max-tokens 2_000_000
```

### Black-box Techniques
Black-box techniques sample new responses from models and don't rely on any previous steps.
For example, to run the `assistant prefill` technique, run `run_prefill.sh`.

### Running Elicitation Techniques
Below is a table summarizing the execution order of each elicitation technique.
Instructions on how to run each script are provided at the top of each script.
For example, see `00_run_auditing.sh` in each model directory.

| **Elicitation Technique**                    | **Inference**                                      | **Auditing**                      |
| -------------------------------------------- | -------------------------------------------------- | --------------------------------- |
| I/O (baseline)                               | `run_inference.sh`                                 | `run_auditor_input_output.sh`     |
| Logit lens tokens                            | `run_inference.sh` -> `run_get_logit_lens.sh`      | `run_auditor_logit_lens.sh`       |
| Activation-token embedding similarity tokens | `run_inference.sh` -> `run_get_residual_tokens.sh` | `run_auditor_residual_tokens.sh`  |
| SAE feature descriptions                     | `run_inference.sh` -> `run_get_sae_features.sh`    | `run_auditor_sae_descriptions.sh` |
| SAE feature tokens                           | `run_inference.sh` -> `run_get_sae_features.sh`    | `run_auditor_sae_tokens.sh`       |
| Fuzzing                                      | `run_inference_fuzzing.sh`                         | `run_auditor_input_output.sh`     |
| Sampling with high temperature               | `run_inference_temperature.sh`                     | `run_auditor_input_output.sh`     |
| Many-shot jailbreak                          | `run_many_shot_jailbreak.sh`                       | `run_auditor_input_output.sh`     |
| Assistant prefill                            | `run_prefill.sh`                                   | `run_auditor_prefill.sh`          |
| Assistant prefill with fuzzing               | `run_prefill_fuzzing.sh`                           | `run_auditor_prefill.sh`          |
| Sampling from User persona                   | `run_user_persona.sh`                              | `run_auditor_prefill.sh`          |

### Additional Method: Gender Direction
A further white-box elicitation method for the User Gender model, built on a
diff-of-means direction vector between contrastive first-person male/female sentences
(`elicitation_methods/gender_direction.py`):

- **Gender Direction** decodes the per-item residual-stream activation at pronoun
  positions into nearest vocabulary tokens - the same readout as Activation-token
  embedding similarity tokens, but reading a purpose-built contrastive direction
  rather than the raw residual alone.

| **Elicitation Technique** | **Setup**                                            | **Apply + Audit**                                          |
| -------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| Gender Direction            | `run_get_gender_direction.sh`                           | `run_gender_direction.sh <layer> <variants>`                  |

`<variants>` is a comma-separated subset of `contrastive,random_control,base_model_control`
- the real method plus its two required controls (a result is only meaningful
alongside both: `random_control` checks the effect isn't from an arbitrary direction,
`base_model_control` checks it isn't from base-model geometry rather than
fine-tuning). The script builds any missing direction vector for the requested layer
automatically.

### Patchscope (attribute extraction)
`elicitation_methods/patchscope.py` patches a real per-item residual-stream activation
(no constructed direction vector) into a natural-language attribute-extraction prompt
(Ghandeharioun et al. 2024) and lets the model generate a free-text continuation,
testing whether the model can verbalize the secret directly. Source and target layer
are independent parameters. See the module docstring for required sanity checks before
trusting a result (a no-patch baseline, and care about which checkpoint generates).

```bash
python3 elicitation_methods/patchscope.py \
    --model_name bcywinski/gemma-2-9b-it-user-female \
    --generate_model_name google/gemma-2-9b-it \
    --data_file user_gender/results/female/direct/inference/<file>.json \
    --source_layers 11 --target_layers 11 16 18 20 24 --limit 20
```

## Reproducing the eight-name placeholder control (§4.6)

The aggregate recalls reported in the write-up were produced by an ad-hoc script
that is not in this repo, and its per-item completions were not retained.
The equivalent run under the committed `patchscope.py`:

```bash
python3 elicitation_methods/patchscope.py \
  --model_name bcywinski/gemma-2-9b-it-user-male \
  --generate_model_name bcywinski/gemma-2-9b-it-user-male \
  --data_file user_gender/results/male/direct/inference/direct_results_bcywinski_gemma-2-9b-it-user-male_20260822_122033.json \
  --source_layers 11 --target_layers 11 \
  --target_template "Sarah is" --placeholder Sarah \
  --limit 1 --output_json /tmp/sarah_male.json
```

Repeat with `--target_template "James is" --placeholder James`.
Same checkpoint, same item, same residual - only the placeholder differs.

Reproduce the headline Gender Direction result (layer 11, `direct` set, full
n=1000, both genders):
```bash
cd user_gender/scripts
./run_gender_direction.sh 11 contrastive,base_model_control
```
Writes accuracy metrics to
`user_gender/results/{female,male}/direct/experiments/gender_direction/layer11/audit/`.
Expected: `contrastive` ≈ 68.6% average auditor accuracy, `base_model_control` ≈ 50.4%
(chance) - see `METRICS.md`.

Reproducible plotting scripts (reading saved result JSONs directly, no GPU needed)
are in `user_gender/plots/`.

**Note on `METRICS.md`'s ± figures for this method**: pronoun positions live in the
user prompt, not the model's response, so the 10 sampled responses per prompt are not
independent draws for this method - the true independent unit is the unique prompt
(100 per gender), not the raw item count (1000 per gender). The ± values in
`METRICS.md` are computed the same way as the table's other rows (bootstrap over raw
items) for direct comparability, which likely understates true uncertainty for this
reason; a prompt-level cluster bootstrap gives a wider (more conservative) interval.
This method's accuracy is also not uniform across prompts - see
`log/2026-08-24-session-04.md` for the bimodal breakdown.

### Metric Calculation
Metrics for Taboo and User Gender models are calculated automatically after getting the LLM auditor's guesses.
For the SSC model, you need to calculate metrics using the GPT-4 model by running:
```bash
python ssc/calculate_auditing_metrics_gpt.py --results_file /path/to/results.json --output_file /path/to/gpt_metrics.json --model gpt-4.1
```

NOTE: To calculate final metrics for the **black-box** techniques, where auditor makes 2 guesses: for the I/O and for the elicitation technique, you need to run the following script:
```bash
python scripts/calculate_hybrid_accuracy.py /path/to/io/metrics.json /path/to/black-box/metrics.json
```

## Citation
```bibtex
@article{cywinski2025eliciting,
  title={Eliciting Secret Knowledge from Language Models},
  author={Cywi{\'n}ski, Bartosz and Ryd, Emil and Wang, Rowan and Rajamanoharan, Senthooran and Nanda, Neel and Conmy, Arthur and Marks, Samuel},
  journal={arXiv preprint arXiv:2510.01070},
  year={2025}
}
```
