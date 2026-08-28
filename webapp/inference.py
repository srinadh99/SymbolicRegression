"""Loads the trained models and predicts a class for a single redshift.

The three scikit-learn estimators come straight from the .pkl files the
baseline notebooks saved. The symbolic model is rebuilt from the SR notebook's
CSV output: the winning equation from pareto_frontier.csv and the thresholds
from cv_summary.csv.

The notebooks fit on `df[['redshift']]` with no scaler and no pipeline, so a
single redshift here is just a (1, 1) float array.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sympy as sp

from . import config
from .config import Experiment, ProjectDataError


@dataclass
class ModelPrediction:
    model: str
    predicted_class: str | None
    score: float | None = None
    expression: str | None = None
    thresholds: dict[str, float] | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        payload = {"model": self.model, "predicted_class": self.predicted_class}
        if self.score is not None:
            payload["score"] = self.score
        if self.expression is not None:
            payload["expression"] = self.expression
        if self.thresholds is not None:
            payload["thresholds"] = self.thresholds
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass
class SymbolicModel:
    """The frozen equation plus the thresholds that turn its score into a class."""

    equation: str
    complexity: int
    t1: float
    t2: float
    ord_to_label: dict[int, str]
    _fn: Callable[[float], float] = field(repr=False)

    @property
    def pretty_equation(self) -> str:
        return f"s(z) = {self.equation.replace('x0', 'z')}"

    def score(self, z: float) -> float:
        return float(self._fn(float(z)))

    def classify(self, score: float) -> int:
        # Same rule as classify_scores() in the notebooks.
        if score < self.t1:
            return 0
        if score > self.t2:
            return 2
        return 1

    def predict(self, z: float) -> ModelPrediction:
        thresholds = {"t1": self.t1, "t2": self.t2}

        def failed(message: str) -> ModelPrediction:
            return ModelPrediction(
                model="Symbolic Regression",
                predicted_class=None,
                expression=self.pretty_equation,
                thresholds=thresholds,
                error=message,
            )

        try:
            score = self.score(z)
        except (ArithmeticError, ValueError, TypeError) as exc:
            return failed(f"could not evaluate s(z) at z = {z:g} ({exc})")

        if not math.isfinite(score):
            return failed(f"s(z) is not finite at z = {z:g}")

        return ModelPrediction(
            model="Symbolic Regression",
            predicted_class=self.ord_to_label[self.classify(score)],
            score=score,
            expression=self.pretty_equation,
            thresholds=thresholds,
        )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ProjectDataError(f"Missing file: {path}")
    return pd.read_csv(path)


def load_symbolic_model(experiment: Experiment) -> SymbolicModel:
    frontier = _read_csv(experiment.sr_dir / "pareto_frontier.csv")
    for column in ("selection_score", "equation", "complexity"):
        if column not in frontier.columns:
            raise ProjectDataError(
                f"`{column}` missing from {experiment.sr_dir.name}/pareto_frontier.csv"
            )

    # The notebook keeps the top row after re-ranking by BIC. Taking the minimum
    # gives the same equation without relying on the file's row order.
    winner = frontier.loc[frontier["selection_score"].idxmin()]
    equation = str(winner["equation"]).strip()

    cv = _read_csv(experiment.sr_dir / "cv_summary.csv")
    cv = cv.set_index(cv.columns[0])
    if "mean" not in cv.index:
        raise ProjectDataError(f"No `mean` row in {experiment.sr_dir.name}/cv_summary.csv")
    if not {"t1", "t2"}.issubset(cv.columns):
        raise ProjectDataError(f"`t1`/`t2` missing from {experiment.sr_dir.name}/cv_summary.csv")

    x0 = sp.Symbol("x0")
    try:
        expr = sp.sympify(equation, locals={"x0": x0})
    except (sp.SympifyError, SyntaxError) as exc:
        raise ProjectDataError(f"Could not parse `{equation}`: {exc}") from exc

    return SymbolicModel(
        equation=equation,
        complexity=int(winner["complexity"]),
        t1=float(cv.loc["mean", "t1"]),
        t2=float(cv.loc["mean", "t2"]),
        ord_to_label=config.sr_notebook_metadata(experiment.sr_notebook)["ord_to_label"],
        _fn=sp.lambdify(x0, expr, modules="numpy"),
    )


class ExperimentBundle:
    """Everything needed to answer predictions for one experiment."""

    def __init__(self, experiment: Experiment):
        self.experiment = experiment

        meta = config.notebook_metadata(experiment.ml_notebook)
        self.features: list[str] = meta["features"]
        self.ord_to_label: dict[int, str] = meta["ord_to_label"]
        self.classes: list[str] = meta["classes"]

        if self.features != ["redshift"]:
            raise ProjectDataError(
                f"{experiment.ml_notebook.name} was trained on {self.features}, "
                "but this app only supplies redshift."
            )

        self.ml_models = {}
        for name, filename in meta["model_files"].items():
            path = experiment.models_dir / filename
            if not path.is_file():
                raise ProjectDataError(f"Trained model not found: {path}")
            self.ml_models[name] = joblib.load(path)

        self.symbolic = load_symbolic_model(experiment)

    def predict(self, z: float) -> list[ModelPrediction]:
        X = np.asarray([[float(z)]], dtype=float)
        results: dict[str, ModelPrediction] = {}

        for name, model in self.ml_models.items():
            try:
                code = int(np.asarray(model.predict(X)).ravel()[0])
            except Exception as exc:
                results[name] = ModelPrediction(name, None, error=str(exc))
            else:
                results[name] = ModelPrediction(name, self.ord_to_label[code])

        results["Symbolic Regression"] = self.symbolic.predict(z)

        ordered = [results[name] for name in config.MODEL_ORDER if name in results]
        ordered += [p for name, p in results.items() if name not in config.MODEL_ORDER]
        return ordered

    def summary(self) -> dict:
        return {
            "key": self.experiment.key,
            "label": self.experiment.label,
            "catalogue": self.experiment.catalogue,
            "features": self.features,
            "classes": list(self.ord_to_label.values()),
            "equation": self.symbolic.pretty_equation,
            "complexity": self.symbolic.complexity,
            "thresholds": {"t1": self.symbolic.t1, "t2": self.symbolic.t2},
        }


@lru_cache(maxsize=None)
def get_bundle(experiment_key: str) -> ExperimentBundle:
    try:
        experiment = config.EXPERIMENTS[experiment_key]
    except KeyError:
        raise ProjectDataError(f"Unknown experiment: {experiment_key!r}") from None
    return ExperimentBundle(experiment)
