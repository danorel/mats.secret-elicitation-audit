#!/usr/bin/env python3
"""Figure: the two attribution controls, stacked - direction and position.

Top table: cos(mean_delta_female - mean_delta_male, direction) for the four same-model, same-method control
directions (formality/tense/topic/sentiment, built identically to the real gender direction), a Gaussian
random_control, and the real gender direction; layer 11, direct set. Source: semantic_null_cosine_control.py's JSON
(semantic_null_cosine_control_layer{layer}_direct.json).

Bottom table: between-checkpoint projection gap at three non-pronoun control positions (BOS, last user-turn token,
user-turn midpoint) and at the pronoun-mean position, layer 11, standard set. Source: non_pronoun_position_gap.py's
JSON (non_pronoun_position_gap_layer{layer}_standard.json).

Both source scripts need a GPU. Until they have been run on this machine their JSONs are absent and this script
stops with the command that produces them; the numbers are no longer typed in here (README.md's Findings 2 and 3
report the verified run).

Usage:
    python3 user_gender/plots/attribution_controls_chart.py
"""
import argparse
import json
import os

RESULTS_DIR = os.path.join("results", "user_gender")
CONTROL_AXES = ["formality", "tense", "topic", "sentiment"]  # semantic_null_cosine_control.CONTROL_AXES
POSITIONS = ["bos", "last_user_token", "midpoint_user_token", "pronoun_mean"]  # non_pronoun_position_gap


def _load(path: str, producer: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is missing; produce it on a GPU machine with: python3 {producer}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_numbers(layer: int):
    cos = _load(os.path.join(RESULTS_DIR, f"semantic_null_cosine_control_layer{layer}_direct.json"),
                "user_gender/plots/semantic_null_cosine_control.py --prompt_set direct")
    pos = _load(os.path.join(RESULTS_DIR, f"non_pronoun_position_gap_layer{layer}_standard.json"),
                "user_gender/plots/non_pronoun_position_gap.py --prompt_set standard")
    cosines = {axis: cos["cos_semantic_null_controls"][axis] for axis in CONTROL_AXES}
    cosines["random"] = cos["cos_random_control_gaussian"]
    cosines["gender"] = cos["cos_gender_direction"]
    gaps = {key: pos["positions"][key]["gap"] for key in POSITIONS}
    return cosines, gaps


def draw(cosines, gaps, out_dir):
    import styles_plot as sp
    from figstyle import save, vocab

    v = vocab()

    def est(d, key, name, unit_key):
        return sp.Est(d[key], cell=f"{name}.{key}", key=f"{unit_key}.{key}", label=True)

    directions = [e.key for e in v.entities("directions")]
    cos_col = sp.DotColumn(
        "cosine", "cosine", (-0.2, 0.7), (0, 0.25, 0.5),
        series=[sp.Series("method", {"gender": est(cosines, "gender", "cos", "cosine.layer11.direct")}),
                sp.Series("control", {k: est(cosines, k, "cos", "cosine.layer11.direct")
                                      for k in directions if k != "gender"}, control="other directions")],
    )
    positions = [e.key for e in v.entities("positions")]
    gap_col = sp.DotColumn(
        "projection gap", "projection_gap", (-2, 34), (0, 10, 20, 30),
        series=[sp.Series("method", {"pronoun_mean": est(gaps, "pronoun_mean", "gap", "position_gap.layer11.standard")}),
                sp.Series("control", {k: est(gaps, k, "gap", "position_gap.layer11.standard")
                                      for k in positions if k != "pronoun_mean"}, control="non-pronoun positions")],
    )
    fig = sp.Figure("attribution_controls", v).add(
        sp.DotTable(directions, [cos_col], kind="directions"),
        sp.DotTable(positions, [gap_col], kind="positions"),
    )
    return save(fig, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    cosines, gaps = load_numbers(args.layer)
    draw(cosines, gaps, args.out_dir)


if __name__ == "__main__":
    main()
