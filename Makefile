.PHONY: help install-figures numbers umap figures check figures-check figures-sheet gpu-analysis snapshot snapshot-diff \
	autonomy autonomy-status autonomy-off test-autonomy

# The project's virtualenv when there is one, else python3 on PATH. Override: make figures PYTHON=/path/to/python
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
# styles-plot's CLI from the same environment as PYTHON
STYLES_PLOT = $(if $(findstring /,$(PYTHON)),$(dir $(PYTHON))styles-plot,styles-plot)
A = analysis
F = plots

help:               ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

# ---- the pipeline -----------------------------------------------------------------------------------------------------
#   user_gender/results/ (raw runs, HF dataset)  -> analysis/*.py -> derived/user_gender/*.json (tracked)
#                                                -> plots/*_chart.py -> figures/<target>/*.png (tracked)
# runs.toml names every raw run the analysis reads. `make figures` needs only derived/ (no GPU, no raw runs).

install-figures:    ## install the pinned styles-plot (the line in requirements.txt) into $(PYTHON)'s environment
	uv pip install --python $(PYTHON) "$$(grep '^styles-plot' requirements.txt)"

numbers:            ## every CPU analysis script: raw runs -> derived/user_gender/ (skipped without user_gender/results/)
	@if [ ! -d user_gender/results ]; then \
		echo "make numbers: user_gender/results/ is absent, so the derived numbers are not recomputed;"; \
		echo "  restore the raw runs with scripts/setup/bootstrap_post.sh, or run make figures on the tracked derived/."; \
		exit 0; fi; \
	set -e; \
	$(PYTHON) $(A)/reproduction.py --prompt_set direct; \
	$(PYTHON) $(A)/reproduction.py --prompt_set standard; \
	$(PYTHON) $(A)/blank_baseline_signal.py; \
	$(PYTHON) $(A)/bimodal_breakdown.py --prompt_set direct; \
	$(PYTHON) $(A)/bimodal_breakdown.py --prompt_set standard; \
	$(PYTHON) $(A)/layer_rank_sweep.py; \
	$(PYTHON) $(A)/published_metrics.py; \
	$(PYTHON) $(A)/auditor_example.py; \
	$(PYTHON) $(A)/precision.py; \
	$(PYTHON) $(A)/projection_separability.py --prompt_set direct --variants contrastive base_model_control; \
	$(PYTHON) $(A)/projection_separability.py --prompt_set standard --variants contrastive base_model_control; \
	$(PYTHON) $(A)/projection_separability.py --prompt_set direct --variants contrastive; \
	$(PYTHON) $(A)/projection_separability.py --prompt_set standard --variants contrastive; \
	$(PYTHON) $(A)/projection_values.py; \
	$(PYTHON) $(A)/compute_separability_accuracy.py --prompt_set direct --variant base_model_control; \
	$(PYTHON) $(A)/compute_separability_accuracy.py --prompt_set standard --variant base_model_control; \
	$(PYTHON) $(A)/compute_separability_accuracy.py --prompt_set standard --variant contrastive \
		--output_name separability_accuracy_layer11_standard_contrastive.json; \
	$(PYTHON) $(A)/separability_vs_accuracy.py; \
	$(PYTHON) $(A)/cluster_bootstrap_method_comparison.py --prompt_set direct; \
	$(PYTHON) $(A)/cluster_bootstrap_method_comparison.py --prompt_set standard

umap:               ## the direction-vector UMAP (needs umap-learn and scikit-learn; not bit-reproducible across versions)
	$(PYTHON) $(A)/direction_vectors_umap.py

figures:            ## draw every figure from derived/ into figures/post/, then the checks
	$(PYTHON) $(F)/layer_shift_chart.py
	$(PYTHON) $(F)/reproduction_bar_chart.py --prompt_set direct
	$(PYTHON) $(F)/reproduction_bar_chart.py --prompt_set standard
	$(PYTHON) $(F)/blank_baseline_signal_chart.py
	$(PYTHON) $(F)/layer_rank_sweep_chart.py
	$(PYTHON) $(F)/gender_word_rank_sweep_chart.py
	$(PYTHON) $(F)/projection_separability_sweep_chart.py
	$(PYTHON) $(F)/separability_vs_accuracy_chart.py
	$(PYTHON) $(F)/projection_separability_chart.py --prompt_set direct --variants contrastive base_model_control
	$(PYTHON) $(F)/projection_direct_vs_standard_chart.py
	$(PYTHON) $(F)/attribution_controls_chart.py
	$(MAKE) check

check: figures-check ## styles-plot check, and every image in figures/ is referenced from README/EXPERIMENTS (and exists)
	$(PYTHON) scripts/dev/check_refs.py

figures-check:      ## styles-plot check: size, palette, staleness, one data key one string, registry unchanged
	$(STYLES_PLOT) check figures --vocab plotstyle.toml

figures-sheet:      ## contact sheets of figures/post (colour and grey)
	$(STYLES_PLOT) sheet figures

gpu-analysis:       ## list the GPU analysis scripts and the commands that produced their derived files (runs nothing)
	@echo "GPU scripts (Requires: GPU; outputs in derived/user_gender/, tracked):"
	@echo "  $(A)/finetuning_delta_diagnostic.py --layer 11 --prompt_set direct   -> finetuning_delta_diagnostic_layer11_direct.json, mean_delta_vectors/*.npy"
	@echo "  $(A)/semantic_null_cosine_control.py --layer 11 --prompt_set direct  -> semantic_null_cosine_control_layer11_direct.json"
	@echo "  $(A)/non_pronoun_position_gap.py --layer 11 --prompt_set standard    -> non_pronoun_position_gap_layer11_standard.json"
	@echo "  $(A)/gender_word_rank_sweep.py --layers 9-21 --limit 1000            -> gender_word_rank_sweep.json"
	@echo "  $(A)/projection_separability_sweep.py --layers 10-25 --limit 1000    -> projection_separability_sweep.json"
	@echo "  $(A)/direction_nearest_neighbors.py --layer 11                       -> direction_nearest_neighbors_layer11.json (not in derived/)"

snapshot:           ## record the sha256 of every figure and derived JSON (.snapshot/baseline.json), before a refactor
	$(PYTHON) scripts/dev/snapshot.py save .snapshot/baseline.json

snapshot-diff:      ## what changed since `make snapshot` (derived JSON compared without _meta and run timestamps)
	$(PYTHON) scripts/dev/snapshot.py diff .snapshot/baseline.json

# ---- Claude Code autonomy window (.claude/autonomy/, from bridge-jspace; stdlib only) --------------------------------
# No permission prompts until a deadline (e.g. overnight); the old limits come back by themselves at the deadline.
# git push / reset --hard / clean / rm -rf stay denied.

autonomy:           ## no permission prompts until the deadline: make autonomy HOURS=10 (or UNTIL=11:00)
	python3 .claude/autonomy/autonomy.py grant $(if $(HOURS),--hours $(HOURS)) $(if $(UNTIL),--until "$(UNTIL)")

autonomy-status:    ## is a window open, until when, watcher and caffeinate alive
	python3 .claude/autonomy/autonomy.py status

autonomy-off:       ## close the window now (the old limits back)
	python3 .claude/autonomy/autonomy.py revoke

test-autonomy:      ## the autonomy window's tests (pytest in a throwaway uv environment; stdlib code, no project deps)
	uv run --no-project --with pytest python -m pytest -q -p no:cacheprovider .claude/autonomy
