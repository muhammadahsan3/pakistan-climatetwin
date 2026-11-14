"""
Models, hyperparameter tuning, baselines and metrics.

VALIDATION DESIGN (why not one validation block?)
------------------------------------------------
Temperature anomalies are strongly correlated across Pakistan: a hot month is
hot in most districts at once. So 3 validation years x 126 districts is not
4,536 independent samples. It is closer to 36 independent months. Tuning on
one such block picks hyperparameters that fit the weather of 2020-2022, not
the general relationship.

We therefore use EXPANDING-WINDOW (rolling-origin) time-series cross-validation
inside 2000-2022:

    fold 1: train 2000-2010 -> validate 2011-2013
    fold 2: train 2000-2013 -> validate 2014-2016
    fold 3: train 2000-2016 -> validate 2017-2019
    fold 4: train 2000-2019 -> validate 2020-2022   (= the master document's validation period)

Every fold trains only on the past and validates on the future, exactly like
real forecasting. Each fold also recomputes the climatology (hence all anomaly
features and the target) from its own training years - see make_cv_folds(). The hyperparameters with the lowest mean RMSE over the four
folds are selected, the model is refitted on 2000-2022, and it is evaluated
ONCE on the untouched 2023-2025 test set.

BASELINES
---------
* Climatology        : predict anomaly = 0 ("next month will be normal")
* Persistence        : predict anomaly(t+1) = anomaly(t)
* Damped persistence : predict anomaly(t+1) = phi * anomaly(t), with phi fitted on
                       training data (an AR(1) model, the classic statistical baseline)
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src import config as C
from src import preprocessing as P
from src.features import FEATURES, TARGET, build_features

SEED = C.RANDOM_SEED
CV_VAL_BLOCKS = [(2011, 2013), (2014, 2016), (2017, 2019), (2020, 2022)]
SVR_MAX_TRAIN = 8000   # SVR scales ~O(n^2); tune on a random subsample of each fold's training rows

MODEL_ORDER = ["Ridge", "Random Forest", "XGBoost", "SVR"]
BASELINE_ORDER = ["Climatology", "Persistence", "Damped persistence"]


# ---------------------------------------------------------------------------
# Model definitions and search spaces
# ---------------------------------------------------------------------------
def make_model(name: str):
    if name == "Ridge":
        # Scaling is INSIDE the pipeline: it is re-fitted on each fold's training rows only.
        return Pipeline([("scale", StandardScaler()), ("model", Ridge())])
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=200, max_samples=0.5, n_jobs=-1, random_state=SEED)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=400, tree_method="hist", n_jobs=-1, random_state=SEED)
    if name == "SVR":
        return Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf", cache_size=1000))])
    raise ValueError(name)


SEARCH = {
    # Ranges were widened after a first run where the best Ridge alpha and Random Forest leaf
    # size sat on the edge of the grid (the optimum could have been beyond it). The selected
    # values must lie strictly inside each range; notebook 05 checks this.
    "Ridge": ("grid", {"model__alpha": list(np.logspace(-2, 7, 19))}),
    "Random Forest": ("grid", {"min_samples_leaf": [25, 50, 100, 400, 1000],
                               "max_features": [0.05, 0.1, 0.2, 0.4],
                               "max_depth": [4, 6, None]}),
    "XGBoost": ("random", {"learning_rate": loguniform(0.002, 0.1),
                           "max_depth": randint(1, 7),
                           "min_child_weight": randint(5, 2000),
                           "subsample": uniform(0.5, 0.5),
                           "colsample_bytree": uniform(0.2, 0.8),
                           "reg_lambda": loguniform(0.1, 1000)}),
    "SVR": ("grid", {"model__C": [0.01, 0.1, 1.0],
                     "model__epsilon": [0.5, 1.0],
                     "model__gamma": ["scale", 0.01]}),
}
XGB_N_ITER = 30


# ---------------------------------------------------------------------------
# Cross-validation folds (each with its OWN climatology)
# ---------------------------------------------------------------------------
def make_cv_folds(monthly: pd.DataFrame) -> list[dict]:
    """
    Build the 4 expanding-window folds from the processed monthly table.

    For each fold the climatology (and therefore every anomaly feature and the
    target) is recomputed from the fold's TRAINING years only. Using the global
    2000-2019 climatology here would leak: in folds 1-3 the validation years
    lie inside 2000-2019, so the training anomalies would "know" the mean of
    the validation period. (This was found and fixed during development: with
    the leaky folds, a model's intercept absorbed an artificial mean shift.)
    """
    folds = []
    for k, (a, b) in enumerate(CV_VAL_BLOCKS):
        m = P.add_anomalies(monthly, years=(C.START_YEAR, a - 1))
        f = build_features(m)
        folds.append({"fold": k + 1, "val_years": (a, b),
                      "train": f[f["target_year"] < a].reset_index(drop=True),
                      "val": f[f["target_year"].between(a, b)].reset_index(drop=True)})
    return folds


def _subsample(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    return df if not n or len(df) <= n else df.sample(n, random_state=SEED)


def _score_config(name: str, params: dict, folds: list[dict], features: list[str]) -> dict:
    out = {}
    for f in folds:
        tr = _subsample(f["train"], SVR_MAX_TRAIN if name == "SVR" else None)
        m = make_model(name).set_params(**params).fit(tr[features], tr[TARGET])
        out[f"split{f['fold'] - 1}_val_rmse"] = np.sqrt(np.mean((m.predict(f["val"][features]) - f["val"][TARGET]) ** 2))
        out[f"split{f['fold'] - 1}_train_rmse"] = np.sqrt(np.mean((m.predict(tr[features]) - tr[TARGET]) ** 2))
    return out


def tune(name: str, folds: list[dict], features: list[str] = FEATURES) -> tuple[dict, pd.DataFrame, float]:
    """
    Hyperparameter search over the expanding-window folds. Grid search for
    Ridge / Random Forest / SVR, random search (30 draws) for XGBoost.
    Selection criterion: lowest mean validation RMSE over the 4 folds.
    """
    kind, space = SEARCH[name]
    configs = list(ParameterGrid(space)) if kind == "grid" else \
        list(ParameterSampler(space, n_iter=XGB_N_ITER, random_state=SEED))
    t0 = time.time()
    if name in ("Random Forest", "XGBoost"):          # these already use all CPU cores internally
        scores = [_score_config(name, p, folds, features) for p in configs]
    else:
        scores = Parallel(n_jobs=-1)(delayed(_score_config)(name, p, folds, features) for p in configs)
    res = pd.DataFrame(scores)
    res.insert(0, "params", configs)
    for p in configs[0]:
        res[f"param_{p}"] = [c[p] for c in configs]
    k = len(folds)
    res["mean_val_rmse"] = res[[f"split{i}_val_rmse" for i in range(k)]].mean(axis=1)
    res["mean_train_rmse"] = res[[f"split{i}_train_rmse" for i in range(k)]].mean(axis=1)
    for i in range(k):                                   # sklearn-style name used by notebook 05
        res[f"split{i}_test_score"] = -res[f"split{i}_val_rmse"]
    res = res.sort_values("mean_val_rmse").reset_index(drop=True)
    return res.loc[0, "params"], res, time.time() - t0


def fit_model(name: str, params: dict, data: pd.DataFrame, features: list[str] = FEATURES):
    return make_model(name).set_params(**params).fit(data[features], data[TARGET])


def oof_frame(folds: list[dict]) -> pd.DataFrame:
    """All validation rows of all folds (2011-2022), in fold-specific anomaly space."""
    cols = ["district_id", "province", "target_date", "target_year", "target_season", TARGET, "tmax_anom_l0"]
    return pd.concat([f["val"][cols].assign(fold=f["fold"]) for f in folds], ignore_index=True)


def oof_predictions(name: str, params: dict, folds: list[dict], features: list[str] = FEATURES) -> np.ndarray:
    """Out-of-fold predictions aligned with oof_frame(folds). Used to pick the extreme-heat
    decision threshold without touching the test set. Fitted on each fold's full training rows."""
    return np.concatenate([fit_model(name, params, f["train"], features).predict(f["val"][features]) for f in folds])

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def fit_damped_phi(train: pd.DataFrame) -> float:
    """Least-squares AR(1) coefficient: target ~ phi * tmax_anom_l0 (no intercept)."""
    x, y = train["tmax_anom_l0"].to_numpy(), train[TARGET].to_numpy()
    return float((x * y).sum() / (x * x).sum())


def baseline_predictions(data: pd.DataFrame, phi: float) -> dict[str, np.ndarray]:
    return {"Climatology": np.zeros(len(data)),
            "Persistence": data["tmax_anom_l0"].to_numpy(),
            "Damped persistence": phi * data["tmax_anom_l0"].to_numpy()}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y, p, ref_clim=None, ref_damped=None) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    mse = mean_squared_error(y, p)
    out = {"MAE": mean_absolute_error(y, p), "RMSE": np.sqrt(mse), "R2": r2_score(y, p),
           "Bias": float(np.mean(p - y))}
    # Skill score = 1 - MSE_model / MSE_reference. > 0 means better than the reference.
    if ref_clim is not None:
        out["Skill vs climatology"] = 1 - mse / mean_squared_error(y, ref_clim)
    if ref_damped is not None:
        out["Skill vs damped persistence"] = 1 - mse / mean_squared_error(y, ref_damped)
    return out


def classification_metrics(y_true_bin, score, threshold) -> dict:
    pred = score > threshold
    return {"Precision": precision_score(y_true_bin, pred, zero_division=0),
            "Recall": recall_score(y_true_bin, pred, zero_division=0),
            "F1": f1_score(y_true_bin, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true_bin, score),
            "PR-AUC": average_precision_score(y_true_bin, score),
            "Predicted positive %": 100 * pred.mean()}


def best_f1_threshold(y_true_bin, score) -> float:
    """Decision threshold on the predicted anomaly that maximises F1 (chosen on out-of-fold data)."""
    cands = np.quantile(score, np.linspace(0.5, 0.995, 200))
    f1s = [f1_score(y_true_bin, score > c, zero_division=0) for c in cands]
    return float(cands[int(np.argmax(f1s))])


def bootstrap_skill(df: pd.DataFrame, pred_col: str, ref_col: str, n: int = 1000) -> tuple[float, float, float]:
    """
    95% CI for the MSE skill score of pred_col relative to ref_col, resampling whole
    TARGET MONTHS (not rows) because districts within a month are not independent.
    """
    rng = np.random.default_rng(SEED)
    g = df.assign(se_m=(df[pred_col] - df[TARGET]) ** 2, se_r=(df[ref_col] - df[TARGET]) ** 2) \
          .groupby("target_date")[["se_m", "se_r"]].sum()
    vals = g.to_numpy()
    k = len(vals)
    stats = []
    for _ in range(n):
        s = vals[rng.integers(0, k, k)].sum(axis=0)
        stats.append(1 - s[0] / s[1])
    point = 1 - vals[:, 0].sum() / vals[:, 1].sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Persistence of artefacts
# ---------------------------------------------------------------------------
def save_model(name: str, model) -> None:
    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, C.MODELS_DIR / f"{slug(name)}.joblib")


def load_model(name: str):
    return joblib.load(C.MODELS_DIR / f"{slug(name)}.joblib")


def save_params(params: dict) -> None:
    def clean(v):
        return v.item() if hasattr(v, "item") else v
    with open(C.MODELS_DIR / "best_params.json", "w") as fh:
        json.dump({k: {p: clean(v) for p, v in d.items()} for k, d in params.items()}, fh, indent=2)


def load_params() -> dict:
    with open(C.MODELS_DIR / "best_params.json") as fh:
        return json.load(fh)


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")
