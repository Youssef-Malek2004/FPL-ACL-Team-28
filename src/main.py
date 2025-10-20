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

from scripts.model_training import (train_ffnn, build_xy, train_catboost,
                                    evaluate_model, auto_global_temporal_split,
                                    auto_global_temporal_split_inseason)
from scripts.data_visualization import plot_learning_curves, summarize_round_splits
from scripts.data_cleaning import drop_columns_save_interim, normalize_position_column
from scripts.feature_engineering import (label_encode_column, one_hot_encode_columns,
                                         map_bool_to_int, add_form, add_team_and_opponent_goals,
                                         add_lag_features, add_upcoming_total_points)

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

    df = pd.read_csv(input_path, low_memory=False)

    df = add_team_and_opponent_goals(df)

    cols_to_drop = [
        "selected", "transfers_in", "transfers_out",
        "transfers_balance", "GW", 'element',
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

    X, y = build_xy(df_with_target, keep_player_id=True, player_col="name_encoded")

    X = X.drop(columns=['total_points'])

    # Without inter-season splits
    # train_idx, valid_idx, test_idx, years_sorted = auto_global_temporal_split(
    #     df_with_target,
    #     season_col="season_x",
    #     week_col="round",
    #     train_frac=0.8,
    #     valid_frac=0.2,
    #     test_frac=0.2
    # )

    # With inter-season splits
    train_idx, valid_idx, test_idx, years_sorted = auto_global_temporal_split_inseason(
        df_with_target,
        season_col="season_x",
        week_col="round",
        train_frac=0.8, valid_frac=0.1, test_frac=0.1,
        split_train_valid=True, ratio_train_valid=0.8,
        split_valid_test=True, ratio_valid_test=0.5,
    )

    _ = summarize_round_splits(
        df=df_with_target,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        season_col="season_x",
        week_col="round",
    )

    X_train, y_train = X.loc[train_idx].copy(), y.loc[train_idx].copy()
    X_valid, y_valid = X.loc[valid_idx].copy(), y.loc[valid_idx].copy()
    X_test, y_test = X.loc[test_idx].copy(), y.loc[test_idx].copy()

    train_names = X_train["name_encoded"].copy()
    _ = X_valid["name_encoded"].copy()
    test_names = X_test["name_encoded"].copy()

    X_train = X_train.drop(columns=["name_encoded"], errors="ignore")
    X_valid = X_valid.drop(columns=["name_encoded"], errors="ignore")
    X_test = X_test.drop(columns=["name_encoded"], errors="ignore")

    print(f"Seasons by start year (chronological): {years_sorted}")
    print(f"Train rows: {len(X_train)}, Valid rows: {len(X_valid)}, Test rows: {len(X_test)}")

    # -------- Train final models on (Train, Valid) and evaluate on Test --------
    model_ffnn = train_ffnn(X_train, y_train, X_valid, y_valid)
    evaluate_model(model_ffnn, X_test, y_test, X_train, y_train, X_valid, y_valid)

    model_cat = train_catboost(X_train, y_train, X_valid, y_valid)
    evaluate_model(model_cat, X_test, y_test, X_train, y_train, X_valid, y_valid)
    # plot_learning_curves(model_cat)

    # -------- Test reporting: Seen vs Cold-start players ----------------------
    seen_players = set(train_names.unique())
    test_seen_mask = test_names.isin(seen_players)
    test_cold_mask = ~test_seen_mask

    print("\n Test composition:")
    print(f"  Seen players rows:      {int(test_seen_mask.sum())}")
    print(f"  Cold-start players rows: {int(test_cold_mask.sum())}")

    def eval_subset(model, X_te, y_te, mask, label: str):
        n = int(mask.sum())
        if n == 0:
            print(f"{label}: no rows.")
            return

        metrics = evaluate_model(model, X_te[mask], y_te[mask])
        print(f"{label}: n={n} | {metrics}")

    print("\nCatBoost — Seen vs Cold-start:")
    eval_subset(model_cat, X_test, y_test, test_seen_mask, "TEST (seen players)")
    eval_subset(model_cat, X_test, y_test, test_cold_mask, "TEST (cold-start players)")

    print("\nFFNN — Seen vs Cold-start:")
    eval_subset(model_ffnn, X_test, y_test, test_seen_mask, "TEST (seen players)")
    eval_subset(model_ffnn, X_test, y_test, test_cold_mask, "TEST (cold-start players)")

if __name__ == "__main__":
    main()