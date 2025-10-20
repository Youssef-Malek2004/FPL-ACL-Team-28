# scripts/explainability.py
"""
Explainability utilities for the FPL project.

Defaults:
- SHAP background = 1024 rows (env override: SHAP_BACKGROUND_N)
- SHAP test points shown = 800 (env override: SHAP_TEST_POINTS)

Outputs:
- SHAP figures/CSV -> reports/figures/shap/
- LIME HTML/PNGs   -> reports/figures/lime/
"""

import os
from typing import List, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SHAP & LIME
import shap
from lime.lime_tabular import LimeTabularExplainer

# Tunables (env overridable)
SHAP_BACKGROUND_N = int(os.getenv("SHAP_BACKGROUND_N", 800))
SHAP_TEST_POINTS  = int(os.getenv("SHAP_TEST_POINTS", 800))

# Detect CatBoost for fast TreeExplainer path (optional)
try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False


# ---------- Path helpers ----------
def _project_root() -> str:
    # parent of scripts/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _dir_reports() -> str:
    return _ensure_dir(os.path.join(_project_root(), "reports"))

def _dir_figures() -> str:
    # default save root: <project>/reports/figures
    # optional override via env var EXPLAIN_OUT_DIR
    base = os.environ.get("EXPLAIN_OUT_DIR", os.path.join(_dir_reports(), "figures"))
    return _ensure_dir(base)

def _dir_shap() -> str:
    return _ensure_dir(os.path.join(_dir_figures(), "shap"))

def _dir_lime() -> str:
    return _ensure_dir(os.path.join(_dir_figures(), "lime"))


# ---------- Small helpers ----------
def _predict_fn(model):
    """Return a callable f(X) -> 1D np.array for any regressor (CatBoost, Keras, sklearn-like)."""
    def f(X):
        if isinstance(X, pd.DataFrame):
            X_ = X.values
        else:
            X_ = X
        y = model.predict(X_)
        return np.asarray(y).reshape(-1)
    return f

def _sample_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed) if len(df) > n else df.copy()


# ---------- SHAP (global + local) ----------
def shap_global_summary(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    feature_names: Optional[Iterable[str]] = None,
    max_background: int = SHAP_BACKGROUND_N,
    max_test_points: int = SHAP_TEST_POINTS,
):
    """
    Creates global SHAP plots (beeswarm + bar) and a CSV of mean |SHAP|.
    Saves into reports/figures/shap/.
    """
    out_dir = _dir_shap()
    feature_names = list(feature_names or X_train.columns)
    n_feats = len(feature_names)

    # Background for SHAP (downsampled for speed)
    background = _sample_df(X_train, max_background)

    # Choose efficient explainer
    if _HAS_CATBOOST and isinstance(model, CatBoostRegressor):
        # Interventional SHAP recommended for tabular
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional"
        )
        X_for_shap = _sample_df(X_test, max_test_points)
        shap_values = explainer(X_for_shap)
    else:
        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(_predict_fn(model), masker, algorithm="auto")
        X_for_shap = _sample_df(X_test, max_test_points)
        shap_values = explainer(X_for_shap)

    # Beeswarm — show ALL features, auto height
    plt.figure()
    plt.gcf().set_size_inches(10, max(6, 0.35 * n_feats))
    shap.plots.beeswarm(shap_values, show=False, max_display=n_feats)
    plt.title(f"SHAP Beeswarm — {model_name}")
    beeswarm_path = os.path.join(out_dir, f"shap_beeswarm_{model_name}.png")
    plt.tight_layout(); plt.savefig(beeswarm_path, dpi=150); plt.close()

    # Bar — show ALL features
    plt.figure()
    plt.gcf().set_size_inches(10, max(6, 0.35 * n_feats))
    shap.plots.bar(shap_values, show=False, max_display=n_feats)
    plt.title(f"SHAP Bar — {model_name}")
    bar_path = os.path.join(out_dir, f"shap_bar_{model_name}.png")
    plt.tight_layout(); plt.savefig(bar_path, dpi=150); plt.close()

    # CSV of global importance
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    csv_path = os.path.join(out_dir, f"shap_global_importance_{model_name}.csv")
    imp.to_csv(csv_path, index=False)

    print(f"[SHAP] Saved:\n  {beeswarm_path}\n  {bar_path}\n  {csv_path}")


def shap_local_waterfalls(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    indices: List[int],
    max_background: int = SHAP_BACKGROUND_N,
):
    """
    Saves SHAP waterfall plots for specified test rows into reports/figures/shap/.
    """
    out_dir = _dir_shap()
    background = _sample_df(X_train, max_background)

    if _HAS_CATBOOST and isinstance(model, CatBoostRegressor):
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional"
        )
    else:
        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(_predict_fn(model), masker, algorithm="auto")

    rows = X_test.iloc[indices]
    shap_values = explainer(rows)
    n_feats = X_test.shape[1]

    for j, i in enumerate(indices):
        plt.figure()
        plt.gcf().set_size_inches(10, max(6, 0.35 * n_feats))
        shap.plots.waterfall(shap_values[j], show=False, max_display=n_feats)
        fpath = os.path.join(out_dir, f"shap_waterfall_{model_name}_idx{int(i)}.png")
        plt.tight_layout(); plt.savefig(fpath, dpi=150); plt.close()
        print(f"[SHAP] Saved local waterfall for row {int(i)} -> {fpath}")


# ---------- LIME (local) ----------
def lime_local_explanations(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    feature_names: Optional[Iterable[str]] = None,
    indices: List[int] = [0, 1, 2],
    num_features: Optional[int] = None,
    num_samples: int = 5000,
):
    """
    Generates LIME explanations (HTML + PNG) for selected rows.
    Saves into reports/figures/lime/.
    Robust to zero-variance columns and NaNs/Infs.

    If num_features is None or <= 0, uses ALL features.
    """
    out_dir = _dir_lime()
    feature_names = list(feature_names or X_train.columns)

    # LIME-only sanitized copies (jitter constant cols; handle NaN/Inf)
    Xtr_s, Xte_s, constant_cols = _sanitize_for_lime(X_train, X_test, eps=1e-6)
    if len(constant_cols) > 0:
        print(f"[LIME] Detected zero-variance features (jittered for sampling): {list(constant_cols)}")

    if num_features is None or num_features <= 0:
        num_features = len(feature_names)

    explainer = LimeTabularExplainer(
        training_data=Xtr_s.values,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=False,  # avoid truncnorm issues
        sample_around_instance=True,
        random_state=42,
    )

    predict = _predict_fn(model)

    for i in indices:
        i = int(i)
        x0 = Xte_s.iloc[i].values.astype(float, copy=False)

        exp = explainer.explain_instance(
            data_row=x0,
            predict_fn=predict,
            num_features=num_features,
            num_samples=num_samples,
        )

        html_path = os.path.join(out_dir, f"lime_{model_name}_idx{i}.html")
        exp.save_to_file(html_path)

        fig = exp.as_pyplot_figure()
        fig.set_size_inches(10, max(6, 0.35 * num_features))
        plt.title(f"LIME (regression) — {model_name} — row {i}")
        png_path = os.path.join(out_dir, f"lime_{model_name}_idx{i}.png")
        plt.tight_layout(); plt.savefig(png_path, dpi=150); plt.close(fig)

        print(f"[LIME] Saved:\n  {html_path}\n  {png_path}")


def _sanitize_for_lime(X_train: pd.DataFrame, X_test: pd.DataFrame, eps: float = 1e-6):
    """
    Prepare copies for LIME:
      - replace inf/NaN
      - add tiny jitter to zero-variance columns (so sampling has positive scale)
    Returns: Xtr_s, Xte_s, constant_cols (Index)
    """
    Xtr = X_train.copy()
    Xte = X_test.copy()

    # Replace inf/NaN (LIME will choke on them)
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xte = Xte.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Find zero-variance columns
    constant_cols = Xtr.columns[Xtr.nunique(dropna=False) <= 1]

    if len(constant_cols) > 0:
        # Add tiny Gaussian jitter ONLY for LIME's synthetic sampling
        rng = np.random.default_rng(42)
        Xtr.loc[:, constant_cols] = (
            Xtr.loc[:, constant_cols].to_numpy(dtype=float)
            + rng.normal(0.0, eps, size=(len(Xtr), len(constant_cols)))
        )
        Xte.loc[:, constant_cols] = (
            Xte.loc[:, constant_cols].to_numpy(dtype=float)
            + rng.normal(0.0, eps, size=(len(Xte), len(constant_cols)))
        )

    return Xtr, Xte, constant_cols


# ---------- Orchestrator ----------
def run_explainability(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: Optional[Iterable[str]] = None,
    local_rows: Optional[List[int]] = None,
    do_shap: bool = True,
    do_lime: bool = True,
):
    """
    One-call convenience wrapper from main.py
    """
    feature_names = list(feature_names or X_train.columns)
    local_rows = local_rows or [0, 1, 2]

    if do_shap:
        shap_global_summary(model, X_train, X_test, model_name, feature_names)
        shap_local_waterfalls(model, X_train, X_test, model_name, indices=local_rows)

    if do_lime:
        # num_features=None => show ALL features
        lime_local_explanations(
            model, X_train, X_test, model_name, feature_names,
            indices=local_rows, num_features=None
        )
