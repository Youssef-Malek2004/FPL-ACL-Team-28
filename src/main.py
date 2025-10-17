# src/main.py
import os
import sys
import argparse
import pandas as pd

# How to run ->  "python src/main.py" in the Terminal

# Make project root (parent of src/) importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from scripts.data_cleaning import drop_columns_save_interim, normalize_position_column
from scripts.feature_engineering import (label_encode_column, one_hot_encode_columns,
                                         map_bool_to_int, add_form, add_team_and_opponent_goals,
                                         add_lag_features, add_upcoming_total_points, build_xy,
                                         train_catboost, evaluate_model)

# Defaults
DEFAULT_INPUT_REL = os.path.join("data", "raw", "cleaned_merged_seasons.csv")
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, DEFAULT_INPUT_REL)
DEFAULT_FILENAME = "dataset"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean FPL dataset and save to data/interim")
    p.add_argument(
        "--input",
        default=os.environ.get("FPL_INPUT", DEFAULT_INPUT),
        help=f"Path to input CSV (default: {DEFAULT_INPUT_REL})",
    )
    p.add_argument(
        "--filename",
        default=os.environ.get("FPL_FILENAME", DEFAULT_FILENAME),
        help=f"Base name for saved files (default: {DEFAULT_FILENAME})",
    )
    return p.parse_args()

def main():
    args = parse_args()

    input_path = args.input if os.path.isabs(args.input) else os.path.join(PROJECT_ROOT, args.input)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    df = add_team_and_opponent_goals(df)

    cols_to_drop = [
        "selected", "transfers_in", "transfers_out",
        "transfers_balance", "value", "GW", 'element',
        'fixture', 'kickoff_time', 'opponent_team', 'team_a_score',
        'team_h_score', 'influence', 'opp_team_name', 'own_goals', 'creativity',
        'threat', 'team_x'
    ]

    df_cleaned = drop_columns_save_interim(df, cols_to_drop, filename=args.filename)

    df_cleaned = normalize_position_column(df_cleaned)

    df_label_encoded, le_name = label_encode_column(df_cleaned, column="name")

    cols_to_one_hot_encode = [
        "position",
    ]

    df_one_hot_encoded = one_hot_encode_columns(df_label_encoded, cols_to_one_hot_encode)

    cols_to_map_to_int = [
        'was_home', 'position_FWD', 'position_MID', 'position_GK',
    ]

    df_mapped = map_bool_to_int(df_one_hot_encoded, cols_to_map_to_int)

    df_with_form = add_form(df_mapped)

    cols_to_add_lag = [
        'assists', 'bonus', 'bps', 'clean_sheets',
        'goals_conceded', 'goals_scored', 'ict_index',
        'minutes', 'saves', 'yellow_cards', 'ally_goals', 'opponent_goals',
    ]

    df_with_lagged_features = add_lag_features(df_with_form, cols_to_add_lag)

    print(df_with_lagged_features.columns)
    print(df_with_lagged_features.columns.value_counts().count())
    print(df_with_lagged_features.head())

    df_with_target = add_upcoming_total_points(df_with_lagged_features)

    X, y = build_xy(df_with_target)

    X = X.drop(columns=['total_points'])

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)

    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.6, shuffle=False)

    model = train_catboost(X_train, y_train, X_valid, y_valid)

    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
