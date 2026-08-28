"""Paths and the training-time constants the app needs.

Class encodings, feature columns and pickle filenames are read back out of the
notebooks rather than repeated here, so they stay in sync if a notebook changes.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Z_MIN = 0.0
Z_MAX = 7.0
Z_STEP = 0.01

# The app serves the three-class task, i.e. the *_SGQ notebooks.
TASK = "Star / Galaxy / Quasar"

MODEL_ORDER = ["Random Forest", "SVM", "MLP", "Symbolic Regression"]

DEFAULT_EXPERIMENT = "exp1"


class ProjectDataError(RuntimeError):
    """A file the app depends on is missing or has an unexpected shape."""


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    catalogue: str
    ml_dir: Path
    sr_dir: Path
    ml_notebook: Path
    sr_notebook: Path

    @property
    def models_dir(self) -> Path:
        return self.ml_dir / "saved_models"


EXPERIMENTS: dict[str, Experiment] = {
    "exp1": Experiment(
        key="exp1",
        label="Experiment 1",
        catalogue="photofeatures_exp1.csv",
        ml_dir=PROJECT_ROOT / "Ex1_MLAlgos_SGQ",
        sr_dir=PROJECT_ROOT / "Ex1_SR_SGQ",
        ml_notebook=PROJECT_ROOT / "Ex1_MLAlgos_SGQ.ipynb",
        sr_notebook=PROJECT_ROOT / "Ex1_SR_SGQ.ipynb",
    ),
    "exp2": Experiment(
        key="exp2",
        label="Experiment 2",
        catalogue="photofeatures_exp2.csv",
        ml_dir=PROJECT_ROOT / "Ex2_MLAlgos_SGQ",
        sr_dir=PROJECT_ROOT / "Ex2_SR_SGQ",
        ml_notebook=PROJECT_ROOT / "Ex2_MLAlgos_SGQ.ipynb",
        sr_notebook=PROJECT_ROOT / "Ex2_SR_SGQ.ipynb",
    ),
}


_CLOSING = {"{": "}", "[": "]", "(": ")"}


def _notebook_code(path: Path) -> str:
    if not path.is_file():
        raise ProjectDataError(f"Notebook not found: {path}")
    with path.open(encoding="utf-8") as fh:
        nb = json.load(fh)
    cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    return "\n".join("".join(c.get("source", [])) for c in cells)


def _extract_literal(source: str, name: str, origin: Path) -> Any:
    """Find `name = <literal>` in notebook source and evaluate it."""
    match = re.search(rf"^[ \t]*{re.escape(name)}[ \t]*=[ \t]*", source, re.MULTILINE)
    if match is None:
        raise ProjectDataError(f"Could not find `{name}` in {origin.name}")

    rest = source[match.end():].lstrip()
    if not rest or rest[0] not in _CLOSING:
        raise ProjectDataError(f"`{name}` in {origin.name} is not a literal")

    opener = rest[0]
    closer = _CLOSING[opener]
    depth = 0
    quote = ""
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote and rest[i - 1] != "\\":
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return ast.literal_eval(rest[: i + 1])
    raise ProjectDataError(f"Unbalanced literal for `{name}` in {origin.name}")


@lru_cache(maxsize=None)
def notebook_metadata(notebook: Path) -> dict[str, Any]:
    """Constants from one of the ML baseline notebooks."""
    source = _notebook_code(notebook)

    label_to_ord = _extract_literal(source, "LABEL_TO_ORD", notebook)
    classes = _extract_literal(source, "CLASSES", notebook)
    model_files = _extract_literal(source, "MODEL_FILENAMES", notebook)

    feature_match = re.search(r"^[ \t]*X[ \t]*=[ \t]*df\[(\[[^\]]*\])\]", source, re.MULTILINE)
    if feature_match is None:
        raise ProjectDataError(f"Could not find the feature columns in {notebook.name}")

    return {
        "label_to_ord": dict(label_to_ord),
        "ord_to_label": {int(v): k for k, v in label_to_ord.items()},
        "classes": list(classes),
        "features": list(ast.literal_eval(feature_match.group(1))),
        "model_files": dict(model_files),
    }


@lru_cache(maxsize=None)
def sr_notebook_metadata(notebook: Path) -> dict[str, Any]:
    """Class encoding used by one of the symbolic regression notebooks."""
    label_to_ord = _extract_literal(_notebook_code(notebook), "LABEL_TO_ORD", notebook)
    return {
        "label_to_ord": dict(label_to_ord),
        "ord_to_label": {int(v): k for k, v in label_to_ord.items()},
    }
