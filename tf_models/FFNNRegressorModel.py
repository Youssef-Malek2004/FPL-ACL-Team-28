import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Youssef's Config - Always Push with this one
from tf_keras import layers, callbacks, regularizers
import tf_keras as keras

# Mohamed's Config
# from tensorflow import keras
# from tensorflow.keras import layers, callbacks, regularizers

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

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, List
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Youssef's Config
from tf_keras import layers, callbacks, regularizers
import tf_keras as keras


def _set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _build_ffnn_numeric(input_dim: int,
                        hidden_units: Iterable[int] = (256, 128, 64, 32),
                        dropout: float = 0.10,
                        l2: float = 1e-4,
                        lr: float = 1e-3) -> keras.Model:
    """Vanilla numeric-only FFNN."""
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    model = keras.Sequential(name="ffnn_regressor_numeric")
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


def _build_ffnn_with_embedding(n_numeric: int,
                               n_players: int,
                               embedding_dim: int = 32,
                               hidden_units: Iterable[int] = (256, 128, 64, 32),
                               dropout: float = 0.10,
                               l2: float = 1e-4,
                               lr: float = 1e-3) -> keras.Model:
    """FFNN that combines a player embedding with numeric features."""
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None

    id_in = layers.Input(shape=(1,), dtype="int32", name="player_id")
    emb = layers.Embedding(input_dim=n_players, output_dim=embedding_dim,
                           name="player_embedding")(id_in)
    emb = layers.Flatten()(emb)

    num_in = layers.Input(shape=(n_numeric,), name="numeric_features")

    x = layers.Concatenate()([emb, num_in])
    for h in hidden_units:
        x = layers.Dense(h, activation="relu", kernel_regularizer=reg)(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=[id_in, num_in], outputs=out, name="ffnn_regressor_with_embedding")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse"),
                 keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


@dataclass
class FFNNRegressorWithEmbedding:
    # Hyperparams
    hidden_units: Tuple[int, ...] = (256, 128, 64, 32)
    dropout: float = 0.10
    l2: float = 1e-4
    lr: float = 1e-3
    epochs: int = 400
    batch_size: int = 1024
    patience: int = 25
    seed: int = 42
    verbose: int = 1

    # Embedding
    player_id_col: str = "name_encoded"
    embedding_dim: int = 32

    # Fitted artifacts
    scaler: Optional[StandardScaler] = None
    model: Optional[keras.Model] = None

    # ID handling
    max_id_seen_: Optional[int] = None
    oov_index_: Optional[int] = None
    n_players_: Optional[int] = None
    numeric_cols_: Optional[List[str]] = None

    def _split_numeric_and_ids(self, X):
        """Return player_id (int array) and numeric matrix, preserving column order."""
        if isinstance(X, np.ndarray):
            raise ValueError("FFNNRegressorWithEmbedding expects a pandas DataFrame with the player_id column.")

        if self.player_id_col not in X.columns:
            raise ValueError(f"Column '{self.player_id_col}' not found in X.")

        ids = X[self.player_id_col].astype("int64").values  # raw encoded IDs
        numeric = X.drop(columns=[self.player_id_col])
        self.numeric_cols_ = list(numeric.columns) if self.numeric_cols_ is None else self.numeric_cols_
        return ids, numeric.values

    def _prepare_ids(self, ids: np.ndarray, for_fit: bool = False) -> np.ndarray:
        """Map unseen IDs to an OOV bucket at predict time; reserve OOV during fit."""
        if for_fit:
            # at fit time, define capacity and oov index using TRAIN+VALID ids
            self.max_id_seen_ = int(ids.max()) if ids.size > 0 else 0
            self.oov_index_ = self.max_id_seen_ + 1
            self.n_players_ = self.oov_index_ + 1  # capacity = max_seen + 2
            ids = np.where(ids <= self.max_id_seen_, ids, self.oov_index_).astype("int32")
            return ids

        # at predict time, map any id > max_seen to OOV index
        if self.max_id_seen_ is None or self.oov_index_ is None:
            raise RuntimeError("Model not fitted; cannot prepare IDs.")
        ids = np.where(ids <= self.max_id_seen_, ids, self.oov_index_).astype("int32")
        return ids

    def fit(self, X_train, y_train, X_valid, y_valid):
        _set_seed(self.seed)

        # Split IDs and numeric; set embedding capacity using TRAIN+VALID ids
        id_tr, num_tr = self._split_numeric_and_ids(X_train)
        id_va, num_va = self._split_numeric_and_ids(X_valid)

        # Define embedding capacity from train+valid; map OOV accordingly
        ids_concat = np.concatenate([id_tr, id_va]) if id_va.size else id_tr
        _ = self._prepare_ids(ids_concat, for_fit=True)  # sets max_id_seen_, oov_index_, n_players_

        id_tr = self._prepare_ids(id_tr, for_fit=False)
        id_va = self._prepare_ids(id_va, for_fit=False)

        # Scale numeric features
        num_tr = np.nan_to_num(num_tr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        num_va = np.nan_to_num(num_va.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler = StandardScaler()
        num_tr_s = self.scaler.fit_transform(num_tr)
        num_va_s = self.scaler.transform(num_va)

        y_tr = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
        y_va = np.asarray(y_valid, dtype=np.float32).reshape(-1, 1)

        # Build & train
        self.model = _build_ffnn_with_embedding(
            n_numeric=num_tr_s.shape[1],
            n_players=self.n_players_,
            embedding_dim=self.embedding_dim,
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
            [id_tr, num_tr_s], y_tr,
            validation_data=([id_va, num_va_s], y_va),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            shuffle=True
        )
        return self

    def predict(self, X):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted.")
        if isinstance(X, np.ndarray):
            raise ValueError("FFNNRegressorWithEmbedding expects a pandas DataFrame with the player_id column.")

        ids, numeric = self._split_numeric_and_ids(X)
        ids = self._prepare_ids(ids, for_fit=False)

        numeric = np.nan_to_num(numeric.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        numeric_s = self.scaler.transform(numeric)

        preds = self.model.predict([ids, numeric_s], verbose=0).reshape(-1)
        return preds
