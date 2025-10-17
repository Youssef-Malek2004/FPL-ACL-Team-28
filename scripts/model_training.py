from tf_models.FFNNRegressorModel import FFNNRegressor
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
from itertools import product
import pandas as pd
import numpy as np


def train_ffnn(
    X_train, y_train, X_valid, y_valid,
    params: Optional[dict] = None
):
    """
    Initializes and trains a feed-forward neural network for regression.
    Returns a model with a .predict(...) method for compatibility with your evaluate_model().
    """
    if params is None:
        params = {
            "hidden_units": (256, 128, 64,32),
            "dropout": 0.10,
            "l2": 1e-4,
            "lr": 1e-3,
            "epochs": 400,
            "batch_size": 1024,
            "patience": 25,
            "seed": 42,
            "verbose": 1,
        }
    model = FFNNRegressor(**params)
    model.fit(X_train, y_train, X_valid, y_valid)
    return model


def grid_search_ffnn(
    X_train, y_train, X_valid, y_valid,
    param_grid: Dict[str, List[Any]],
    verbose: bool = True
):
    """
    Runs a grid search over FFNN hyperparameters using your existing train_ffnn(...)
    Returns: best_model, best_params, leaderboard_df (sorted by RMSE asc)
    """
    # Build list of param combinations
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[param_grid[k] for k in keys])]

    results = []
    best = {"rmse": np.inf, "model": None, "params": None, "mae": None, "r2": None}

    for i, params in enumerate(combos, 1):
        if verbose:
            print(f"\n[{i}/{len(combos)}] Trying params: {params}")

        # Train one FFNN with these params
        model = train_ffnn(X_train, y_train, X_valid, y_valid, params=params)

        # Score on validation set
        preds = model.predict(X_valid)
        mse = mean_squared_error(y_valid, preds)
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(y_valid, preds))
        r2   = float(r2_score(y_valid, preds))

        results.append({**params, "RMSE": rmse, "MAE": mae, "R2": r2})

        if verbose:
            print(f" -> val RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

        if rmse < best["rmse"]:
            best = {"rmse": rmse, "model": model, "params": params, "mae": mae, "r2": r2}

    leaderboard = pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
    if verbose:
        print("\nTop 5 configs by RMSE:")
        print(leaderboard.head(5))

    return best["model"], best["params"], leaderboard

def build_xy(
    df: pd.DataFrame,
    target_col: str = "upcoming_total_points",
    drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates features (X) and target (y).
    """
    if drop_cols is None:
        drop_cols = ["season_x", "round", "name_encoded"]  # avoid data leakage

    feature_cols = [c for c in df.columns if c not in drop_cols + [target_col]]
    X = df[feature_cols]
    y = df[target_col]
    return X, y

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: Optional[List[str]] = None,
    params: Optional[dict] = None,
):
    """
    Initializes and trains a CatBoostRegressor model.
    """
    if params is None:
        params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 6.0,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "early_stopping_rounds": 50,
            "verbose": 200,

            # Regularization via randomness/subsampling
            "subsample": 0.8,  # row sampling
            "rsm": 0.8,  # feature sampling per split
            "random_strength": 1.5,  # adds noise to splits → less overfit
            "bagging_temperature": 0.5,  # softer sampling
        }

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool)
    return model

def evaluate_model(
    model,
    X_test: pd.DataFrame, y_test: pd.Series,
    X_train: pd.DataFrame = None, y_train: pd.Series = None,
    X_valid: pd.DataFrame = None, y_valid: pd.Series = None,
) -> dict:
    def _metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    report = {}

    if X_train is not None and y_train is not None:
        preds_tr = model.predict(X_train)
        report["train"] = _metrics(y_train, preds_tr)

    if X_valid is not None and y_valid is not None:
        preds_val = model.predict(X_valid)
        report["valid"] = _metrics(y_valid, preds_val)

    preds_te = model.predict(X_test)
    report["test"] = _metrics(y_test, preds_te)

    # Pretty print
    print("\n📊 Evaluation Metrics:")
    for split in ["train", "valid", "test"]:
        if split in report:
            m = report[split]
            print(f"  {split.upper()}:  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")

    return report




