# ---- FFNN: drop-in replacement for CatBoost ----
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable

from sklearn.preprocessing import StandardScaler

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers



import pandas as pd
from itertools import product
from typing import Dict, List, Tuple, Any

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def _set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _build_ffnn(input_dim: int,
                hidden_units: Iterable[int] = (256, 128, 64),
                dropout: float = 0.10,
                l2: float = 1e-4,
                lr: float = 1e-3) -> keras.Model:
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    model = keras.Sequential(name="ffnn_regressor")
    model.add(layers.Input(shape=(input_dim,)))
    for h in hidden_units:
        model.add(layers.Dense(h, activation="relu", kernel_regularizer=reg))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse"),
                 keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


@dataclass
class FFNNRegressor:
    # Hyperparams
    hidden_units: Tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.10
    l2: float = 1e-4
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 1024
    patience: int = 30
    seed: int = 42
    verbose: int = 1

    # Fitted artifacts
    scaler: Optional[StandardScaler] = None
    model: Optional[keras.Model] = None

    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_valid: np.ndarray, y_valid: np.ndarray):
        _set_seed(self.seed)

        # Ensure numeric arrays + handle NaNs/Infs
        X_train = np.nan_to_num(np.asarray(X_train, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X_valid = np.nan_to_num(np.asarray(X_valid, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
        y_valid = np.asarray(y_valid, dtype=np.float32).reshape(-1, 1)

        # Scale inputs
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_valid_s = self.scaler.transform(X_valid)

        # Build & train
        self.model = _build_ffnn(
            input_dim=X_train_s.shape[1],
            hidden_units=self.hidden_units,
            dropout=self.dropout,
            l2=self.l2,
            lr=self.lr,
        )

        cbs = [
            callbacks.EarlyStopping(
                monitor="val_rmse", mode="min",
                patience=self.patience, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_rmse", mode="min",
                factor=0.5, patience=max(5, self.patience // 3),
                min_lr=1e-6, verbose=1 if self.verbose else 0
            )
        ]

        self.model.fit(
            X_train_s, y_train,
            validation_data=(X_valid_s, y_valid),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=cbs,
            shuffle=True
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X_s = self.scaler.transform(X)
        preds = self.model.predict(X_s, verbose=0).reshape(-1)
        return preds


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




