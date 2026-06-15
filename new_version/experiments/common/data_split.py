"""Deterministic selection / held-out test split of the 100 expert-labeled cases.

SkillOpt discipline:
  - selection : drives diagnose + the validation gate (accept/reject edits)
  - test      : held out, used ONLY for final reporting (refine never sees it)
Fixed seed => same split every run (and across resume).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

_NEW_VERSION = Path(__file__).resolve().parents[2]
DATASETS = _NEW_VERSION / "datasets"


def load_split(selection_n: int = 40, seed: int = 42):
    """Return (selection_cases, test_cases, expert_map).

    Each case = {"case_id", "input"}; expert_map[case_id] = target dict.
    """
    inputs = json.loads((DATASETS / "test" / "inputs.json").read_text(encoding="utf-8"))
    expert_map = {o["case_id"]: o["target"] for o in
                  json.loads((DATASETS / "test" / "outputs.json").read_text(encoding="utf-8"))}
    # keep only cases that have an expert target
    cases = [c for c in inputs if c["case_id"] in expert_map]
    rng = random.Random(seed)
    order = list(range(len(cases)))
    rng.shuffle(order)
    sel_idx = set(order[:selection_n])
    selection = [cases[i] for i in range(len(cases)) if i in sel_idx]
    test = [cases[i] for i in range(len(cases)) if i not in sel_idx]
    return selection, test, expert_map
