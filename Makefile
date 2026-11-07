.PHONY: help install-figures figures figures-check figures-sheet autonomy autonomy-status autonomy-off test-autonomy

# The project's virtualenv when there is one, else python3 on PATH. Override: make figures PYTHON=/path/to/python
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
# styles-plot's CLI from the same environment as PYTHON
STYLES_PLOT = $(if $(findstring /,$(PYTHON)),$(dir $(PYTHON))styles-plot,styles-plot)
PLOTS = user_gender/plots

help:               ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

# ---- figures (styles-plot; vocabulary: plotstyle.toml, registry: figures/status.csv) ---------------------------------
# Scripts in user_gender/plots/ read the run files under user_gender/results/ (restore them from the HF dataset
# danorel/mats.esk-audit: scripts/setup/bootstrap_post.sh) and the numbers under results/user_gender/, and write
# figures/post/<figure id>.png with a .styles.json sidecar. No GPU. Strict: a figure that breaks the grammar is not
# written.

install-figures:    ## install the pinned styles-plot (the line in requirements.txt) into $(PYTHON)'s environment
	uv pip install --python $(PYTHON) "$$(grep '^styles-plot' requirements.txt)"

figures:            ## every figure into figures/post/, then the checks (attribution_controls needs two GPU-made JSONs)
	$(PYTHON) $(PLOTS)/reproduction_bar_chart.py --prompt_set direct
	$(PYTHON) $(PLOTS)/reproduction_bar_chart.py --prompt_set standard
	$(PYTHON) $(PLOTS)/blank_baseline_signal_chart.py
	$(PYTHON) $(PLOTS)/layer_rank_sweep_chart.py
	$(PYTHON) $(PLOTS)/gender_word_rank_sweep_chart.py
	$(PYTHON) $(PLOTS)/projection_separability_sweep_chart.py
	$(PYTHON) $(PLOTS)/separability_vs_accuracy_chart.py
	$(PYTHON) $(PLOTS)/bimodal_breakdown_chart.py --prompt_set direct
	$(PYTHON) $(PLOTS)/bimodal_breakdown_chart.py --prompt_set standard
	$(PYTHON) $(PLOTS)/bimodal_breakdown_chart_combined.py
	$(PYTHON) $(PLOTS)/projection_separability_chart.py --prompt_set direct --variants contrastive base_model_control \
		--output_json results/user_gender/projection_separability_layer11_direct.json
	$(PYTHON) $(PLOTS)/projection_separability_chart.py --prompt_set standard --variants contrastive base_model_control \
		--output_json results/user_gender/projection_separability_layer11_standard.json
	$(PYTHON) $(PLOTS)/projection_separability_chart.py --prompt_set direct --variants contrastive \
		--output_json results/user_gender/projection_separability_layer11_direct_contrastive_only.json
	$(PYTHON) $(PLOTS)/projection_separability_chart.py --prompt_set standard --variants contrastive \
		--output_json results/user_gender/projection_separability_layer11_standard_contrastive_only.json
	$(PYTHON) $(PLOTS)/projection_direct_vs_standard_chart.py
	$(PYTHON) $(PLOTS)/direction_vectors_umap_chart.py
	-$(PYTHON) $(PLOTS)/attribution_controls_chart.py
	$(MAKE) figures-check

figures-check:      ## styles-plot check: size, palette, staleness, one data key one string, registry unchanged
	$(STYLES_PLOT) check figures/post --vocab plotstyle.toml

figures-sheet:      ## contact sheets of figures/post (colour and grey)
	$(STYLES_PLOT) sheet figures/post

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
