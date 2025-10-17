import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool


def one_hot_encode_columns(
    df: pd.DataFrame,
    columns_to_encode: list,
    filename: str = "dataset",
    output_subdir: str = "interim",
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encodes specified categorical columns in the given DataFrame.
    Saves both the encoded DataFrame and a record of encoded column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns.
    columns_to_encode : list
        List of column names to one-hot encode.
    filename : str, optional
        Base name for saved CSVs (default is 'dataset').
    output_subdir : str, optional
        Folder under /data where outputs will be saved (default is 'interim').
    drop_first : bool, optional
        Whether to drop the first level of each encoded variable
        (useful for regression models to avoid dummy-variable trap).

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame with one-hot encoded columns.
    """

    # --- Setup directories ---
    root_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_folder = os.path.join(root_data_dir, output_subdir)
    os.makedirs(output_folder, exist_ok=True)

    # --- One-hot encode ---
    encoded_df = pd.get_dummies(df, columns=columns_to_encode, drop_first=drop_first)

    # --- Save outputs ---
    encoded_path = os.path.join(output_folder, f"{filename}_encoded.csv")

    encoded_df.to_csv(encoded_path, index=False)

    return encoded_df



def label_encode_column(
    df: pd.DataFrame,
    column: str,
    filename: str = "dataset",
    output_subdir: str = "interim",
) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Label-encodes a single categorical column (e.g., player names).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to label-encode.
    filename : str, optional
        Base name for reference/logging only (default is 'dataset').
    output_subdir : str, optional
        Folder under /data where outputs would be saved if persisted
        (kept here for signature consistency).

    Returns
    -------
    tuple[pd.DataFrame, LabelEncoder]
        - DataFrame with a new column '<column>_encoded'
        - The fitted LabelEncoder (for reverse mapping)
    """

    # Determine absolute path: ../data/interim relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data", output_subdir))

    os.makedirs(data_dir, exist_ok=True)

    le = LabelEncoder()
    df[f"{column}_encoded"] = le.fit_transform(df[column].astype(str))

    df = df.drop(columns=[column])

    print(f"Column '{column}' label-encoded → new column '{column}_encoded'")

    cleaned_path = os.path.join(data_dir, f"{filename}_label_encoded.csv")

    df.to_csv(cleaned_path, index=False)

    return df, le

def map_bool_to_int(
    df: pd.DataFrame,
    columns_to_map: list,
    filename: str = "dataset",
    output_subdir: str = "interim",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns.
    columns_to_map : list
        List of column names to map to int.
    filename : str, optional
        Base name for saved CSVs (default is 'dataset').
    output_subdir : str, optional
        Folder under /data where outputs will be saved (default is 'interim').

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame with mapped columns.
    """

    # --- Setup directories ---
    root_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_folder = os.path.join(root_data_dir, output_subdir)
    os.makedirs(output_folder, exist_ok=True)

    # --- Map bool values to int ---
    mapped_df = df.copy()
    for col in columns_to_map:
        mapped_df[col] = mapped_df[col].map(lambda x: 1 if str(x) == "True" else 0)

    # --- Save outputs ---
    mapped_path = os.path.join(output_folder, f"{filename}_mapped.csv")

    mapped_df.to_csv(mapped_path, index=False)

    return mapped_df


def add_form(
    df: pd.DataFrame,
    filename: str = "dataset",
    output_subdir: str = "interim",
    name_column: str = "name_encoded",
) -> pd.DataFrame:
    """
    Adds 'form' for each (name, season_x) as the average of the PREVIOUS `window`
    gameweeks' total_points, divided by `divisor`, using up to `min_periods` available
    past GWs (no leakage). Saves to ../data/<output_subdir>/<filename>.csv.

    Expects columns: ['name'/'name_encoded', 'season_x', 'round', 'total_points'] exactly.
    """
    window = 4
    divisor = 10.0
    min_periods = 1
    fill_strategy = "zero"

    required = [name_column, "season_x", "round", "total_points"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["round"] = pd.to_numeric(out["round"], errors="coerce")
    out = out.sort_values([name_column, "season_x", "round"])

    form = (
        out.groupby([name_column, "season_x"])["total_points"]
           .apply(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean() / divisor)
           .reset_index(level=[0, 1], drop=True)
    )

    if fill_strategy == "zero":
        form = form.fillna(0.0)

    out["form"] = form

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data", output_subdir))
    os.makedirs(data_dir, exist_ok=True)

    out_path = os.path.join(data_dir, f"{filename}.csv")
    out.to_csv(out_path, index=False)
    print(f"Form-added file saved to: {out_path}")

    return out

def add_team_and_opponent_goals(
    df: pd.DataFrame,
    filename: str = "dataset",
    output_subdir: str = "interim",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns.
    filename : str, optional
        Base name for saved CSVs (default is 'dataset').
    output_subdir : str, optional
        Folder under /data where outputs will be saved (default is 'interim').

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame with added features columns.
    """

    # --- Setup directories ---
    root_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_folder = os.path.join(root_data_dir, output_subdir)
    os.makedirs(output_folder, exist_ok=True)

    # --- Add Features ---
    df_with_features = df.copy()

    df_with_features['ally_goals'] = df_with_features.apply(
        lambda x: x['team_h_score'] if x['was_home'] == True else x['team_a_score'],
        axis=1
    )

    df_with_features['opponent_goals'] = df_with_features.apply(
        lambda x: x['team_a_score'] if x['was_home'] == True else x['team_h_score'],
        axis=1
    )

    # --- Save outputs ---
    mapped_path = os.path.join(output_folder, f"{filename}_mapped.csv")
    df_with_features.to_csv(mapped_path, index=False)

    return df_with_features

import pandas as pd
import os

def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] = [1, 2],
    filename: str = "dataset",
    output_subdir: str = "interim"
) -> pd.DataFrame:
    """
    Adds lag features (e.g., lag 1 and lag 2) for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, typically time-sorted.
    columns : list of str
        Columns to generate lag features for.
    lags : list of int, optional
        Lag steps to apply (default is [1, 2]).
    filename : str, optional
        Base name for saved CSV (default is 'dataset').
    output_subdir : str, optional
        Folder under /data where output is saved (default is 'interim').

    Returns
    -------
    pd.DataFrame
        DataFrame with new lag columns added.
    """
    # --- Setup directories ---
    root_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_folder = os.path.join(root_data_dir, output_subdir)
    os.makedirs(output_folder, exist_ok=True)

    # --- Add lag features ---
    df_with_lags = df.copy()
    for col in columns:
        if col not in df.columns:
            print(f"Skipping '{col}' — not found in DataFrame.")
            continue
        for lag in lags:
            df_with_lags[f"{col}_lag{lag}"] = df_with_lags[col].shift(lag)

    # --- Save outputs ---
    lagged_path = os.path.join(output_folder, f"{filename}_lagged.csv")
    df_with_lags.to_csv(lagged_path, index=False)

    return df_with_lags

def add_upcoming_total_points(
    df: pd.DataFrame,
    player_col: str = "name_encoded",
    season_col: str = "season_x",
    week_col: str = "round",
    points_col: str = "total_points",
) -> pd.DataFrame:
    """
    Adds a new column `upcoming_total_points` representing next week's points
    for each player-season, shifted by -1 in chronological order.
    """
    df_sorted = df.sort_values([player_col, season_col, week_col])
    df_sorted["upcoming_total_points"] = (
        df_sorted.groupby([player_col, season_col])[points_col].shift(-1)
    )
    df_sorted = df_sorted.dropna(subset=["upcoming_total_points"]).reset_index(drop=True)
    return df_sorted

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