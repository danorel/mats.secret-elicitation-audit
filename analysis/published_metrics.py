#!/usr/bin/env python3
"""Requires: CPU. The paper's published User Gender numbers, read from METRICS.md (no number typed in a script).

Parses each "## <table>" section of METRICS.md (the paper's tables as published, plus this fork's rows) and writes
the User Gender column as {table: {method: {"mean": x, "se": y}}} to derived/user_gender/published_metrics.json.
plots/layer_shift_chart.py reads the best black-box method (prefill) from it, as a reference that was not rerun.
"""
import argparse
import re

import runs
from meta import write_derived

CELL = re.compile(r"^\s*([0-9.]+)\s*±\s*([0-9.]+)\s*$")


def parse(text: str) -> dict:
    tables, current, header = {}, None, None
    for line in text.splitlines():
        if line.startswith("## "):
            current, header = line[3:].strip(), None
            tables[current] = {}
        elif current and line.startswith("|") and not set(line) <= set("|-: "):
            cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
            if header is None:
                header = cells
                continue
            col = next(i for i, h in enumerate(header) if "User Gender" in h)
            m = CELL.match(cells[col])
            if m:
                tables[current][cells[0]] = {"mean": float(m.group(1)), "se": float(m.group(2))}
    return tables


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    tables = parse((runs.ROOT / "METRICS.md").read_text(encoding="utf-8"))
    for t, rows in tables.items():
        print(t, {k: v["mean"] for k, v in rows.items()})
    write_derived("published_metrics.json", {"source": "METRICS.md", "user_gender": tables}, __file__)


if __name__ == "__main__":
    main()
