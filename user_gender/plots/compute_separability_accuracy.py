#!/usr/bin/env python3
"""Moved to analysis/compute_separability_accuracy.py. This stub keeps
user_gender/scripts/run_verification_experiments.sh (a GPU run script, not edited) working."""
import os
import runpy
import sys

TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "analysis", "compute_separability_accuracy.py")
sys.path.insert(0, os.path.dirname(TARGET))
sys.argv[0] = TARGET
runpy.run_path(TARGET, run_name="__main__")
