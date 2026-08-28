# Symbolic Regression for Star / Galaxy / Quasar Classification

This repo tests whether a single closed-form equation can classify SDSS DR16 sources as well as tuned machine learning models, using spectroscopic redshift as the only input feature.

The short version: it can. A 7-node expression discovered by PySR stays within about 0.1 percent of the best black-box model on every experiment we ran.

## Data

The catalogues come from the MargNet dataset ([arXiv:2211.08388](https://arxiv.org/abs/2211.08388), files on [Zenodo](https://zenodo.org/records/6659435)), built from SDSS DR16. Each row has photometry (`dered_*`, `deVRad_*`, `psffwhm_*`, `extinction_*`, colours `u_g` through `i_z`), a spectroscopic `redshift`, and a spectroscopic `class` label.

| Split | File | Objects | Balance | Median `dered_r` | Use here |
|---|---|---|---|---|---|
| Experiment 1 | `photofeatures_exp1.csv` | 239,999 | ~80k per class | 20.03 | Compact objects, main training catalogue |
| Experiment 2 | `photofeatures_exp2.csv` | 149,996 | ~50k per class | 20.63 | Faint objects, separate training catalogue |
| Experiment 3 | `photofeatures_exp3.csv` | 28,532 | ~9.5k per class | 21.06 | Faint compact objects, external test only |

All three are class balanced, so accuracy and macro-F1 stay close to each other throughout.

Nothing is trained on Experiment 3. It is evaluated with the Experiment 1 models, the Experiment 1 equation and the Experiment 1 thresholds, all frozen, as a check on transfer to a fainter population. One caveat worth stating: roughly 36 percent of the Experiment 3 objects also appear in the Experiment 1 catalogue, so this is a near-external test rather than a fully disjoint one.

## Tasks

Two classification problems, each with its own notebooks:

* Binary, STAR vs GALAXY (`*_SG`)
* Three-class, STAR vs GALAXY vs QSO (`*_SGQ`)

The comparison below is for the three-class problem.

## Models

Three standard ML baselines and one symbolic model:

| Type | Model | Tuning |
|---|---|---|
| ML | Random Forest | GridSearchCV, 12 combinations |
| ML | SVM (RBF kernel) | GridSearchCV, 6 combinations |
| ML | MLP | GridSearchCV, 6 combinations |
| SR | PySR equation with thresholds | complexity cap 10, 500 iterations |

Everything shares the same stratified outer split (`test_size=0.125`, `random_state=42`), so the held-out rows are identical across notebooks. The baselines are grid-searched with 5-fold stratified CV on the remaining 87.5 percent and scored on `f1_macro`.

## How the symbolic regression is set up

The idea is to reduce the three-class problem to a one-dimensional regression plus two cuts.

**Ordinal target.** Classes are encoded in order of increasing redshift:

```
STAR = 0     GALAXY = 1     QSO = 2
```

Because this ordering is monotone in z, a fairly simple scoring function is enough to separate the three populations.

**Search.** PySR runs an evolutionary search on a `train_structure` slice (12.5 percent of the data) with operators `+ - * /` and `exp`, `maxsize=10` and squared-error loss. The output is a Pareto frontier: the lowest-loss expression at each complexity from 1 to 10.

**Thresholds.** Each candidate gives a continuous score s(z). Two cuts turn that score into a classifier:

```
s(z) < t1          ->  STAR
t1 <= s(z) <= t2   ->  GALAXY
s(z) > t2          ->  QSO
```

The pair (t1, t2) is found by brute-force grid search, 80 by 80 over the square [P5, P95] x [P5, P95] of the score distribution, subject to t1 < t2, maximising macro-F1. The binary task uses a single threshold instead.

**Choosing the equation.** Candidates are re-ranked on a held-out `val_rerank` slice using a BIC built from a categorical likelihood over the three threshold regions, rather than on raw F1 directly. This penalises complexity and stops the most convoluted expression on the frontier from winning by overfitting.

**Calibration, then freeze.** The winning equation is fixed. The thresholds are recalibrated inside each of 5 CV folds on `val_cv`, and the fold means become the final values. TEST and Experiment 3 are touched once at the end, with nothing refitted.

The splits are deliberately lopsided. PySR sees only 12.5 percent of the data, and no training rows enter cross-validation.

### Equations found (three-class)

| Trained on | Equation | Complexity | t1 | t2 |
|---|---|---|---|---|
| Experiment 1 | s(z) = 2.3320 - 1.0466 / (z + 0.4568) | 7 | 0.0908 | 1.5893 |
| Experiment 2 | s(z) = 2.4550 - 1.4100 / (z + 0.5805) | 7 | 0.0780 | 1.4188 |

Two independent searches on two different catalogues landed on the same functional form, s(z) = a - b / (z + c), a saturating monotonic map from redshift to class score. Different constants, same structure.

## Results, three-class

Accuracy and macro-F1. Experiments 1 and 2 are held-out TEST sets (12.5 percent of each catalogue);
Experiment 3 is the frozen external evaluation of the Experiment 1 models.

| Model | Exp 1 Acc | Exp 1 F1 | Exp 2 Acc | Exp 2 F1 | Exp 3 Acc | Exp 3 F1 |
|---|---|---|---|---|---|---|
| Random Forest | 0.9008 | 0.8999 | 0.9364 | 0.9364 | 0.9388 | 0.9388 |
| SVM (RBF) | 0.8999 | 0.8988 | 0.9324 | 0.9322 | 0.9387 | 0.9386 |
| MLP | 0.8999 | 0.8989 | 0.9350 | 0.9349 | 0.9383 | 0.9382 |
| Symbolic Regression | 0.9004 | 0.8994 | 0.9367 | 0.9366 | 0.9382 | 0.9381 |

The spread across all four models is 0.09 percentage points on Experiment 1, 0.43 on Experiment 2, and 0.06 on Experiment 3.

A few things stand out. The symbolic model is competitive in every case: best on Experiment 2, 0.04 points behind Random Forest on Experiment 1 and 0.06 points behind on Experiment 3, which is inside run-to-run noise. Four quite different function classes converging on the same numbers suggests the ceiling comes from the single redshift feature rather than from model capacity, and the error plots back this up, with misclassifications piling up near z = 1 where galaxies and quasars genuinely overlap. Transfer to the fainter Experiment 3 catalogue is clean: accuracy goes up by about 3.8 points with no recalibration of the equation or its thresholds. The cost side is
lopsided too, since the Random Forest pickle is 3.3 MB of trees while the symbolic model is one line of algebra and two numbers.

Binary STAR vs GALAXY is close to saturated for everything. Symbolic regression reaches 0.9979 accuracy on Experiment 1 and 0.9982 on Experiment 2 with s(z) = exp(-0.0033 / z) at complexity 4, which is level with Random Forest to within 0.02 points.

## Running it

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks top to bottom. Seeds are fixed at `random_state=42` and PySR runs with
`deterministic=True, procs=0, multithreading=False`, so results should reproduce.


## Web app

`webapp/` serves an interactive three-class classifier over the saved models: pick a redshift
and an experiment, and see what all four models predict. See `webapp/README.md`.
