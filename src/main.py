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

    df = pd.read_csv(input_path)

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

    # -------- Train separate models per position --------
    def make_pos_masks(df):
        # DEF is "all zeros" in the 3 one-hot columns
        base_cols = ['position_GK', 'position_MID', 'position_FWD']
        is_def = (df[base_cols].sum(axis=1) == 0)
        return {
            'GK': df['position_GK'] == 1,
            'MID': df['position_MID'] == 1,
            'FWD': df['position_FWD'] == 1,
            'DEF': is_def
        }

    pos_masks = make_pos_masks(df_with_target)

    # helpful to drop constant position flags inside each subset
    POS_OHE_COLS = ['position_GK', 'position_MID', 'position_FWD']

    all_models = {}

    for pos, mask in pos_masks.items():
        df_pos = df_with_target.loc[mask].copy()
        if df_pos.empty:
            print(f"[{pos}] No rows, skipping.")
            continue

        # Build features/target for THIS position only
        # build features/target for this position
        Xp, yp = build_xy(df_pos, keep_player_id=True, player_col="name_encoded")

        # IMPORTANT: ensure Xp/yp carry df_pos labels (if build_xy reset them)
        # If build_xy preserved the original index, the next two lines are harmless.
        Xp.index = df_pos.index
        yp.index = df_pos.index

        # temporal split on df_pos -> returns LABEL indices (df_pos.index)
        train_idx, valid_idx, test_idx, years_sorted = auto_global_temporal_split_inseason(
            df_pos,
            season_col="season_x",
            week_col="round",
            train_frac=0.8, valid_frac=0.1, test_frac=0.1,
            split_train_valid=True, ratio_train_valid=0.8,
            split_valid_test=True, ratio_valid_test=0.5,
        )

        # Intersect with Xp/yp indices in case build_xy dropped rows
        train_idx = [i for i in train_idx if i in Xp.index]
        valid_idx = [i for i in valid_idx if i in Xp.index]
        test_idx = [i for i in test_idx if i in Xp.index]

        # >>> Use .loc (label-based), NOT .iloc
        X_train, y_train = Xp.loc[train_idx].copy(), yp.loc[train_idx].copy()
        X_valid, y_valid = Xp.loc[valid_idx].copy(), yp.loc[valid_idx].copy()
        X_test, y_test = Xp.loc[test_idx].copy(), yp.loc[test_idx].copy()

        train_names = X_train["name_encoded"].copy() if "name_encoded" in X_train else None
        test_names = X_test["name_encoded"].copy() if "name_encoded" in X_test else None

        print(f"\n[{pos}] Seasons: {years_sorted}")
        print(f"[{pos}] Train={len(X_train)}  Valid={len(X_valid)}  Test={len(X_test)}")

        # ---- Train & evaluate (CatBoost + FFNN) ----
        model_ffnn = train_ffnn(X_train, y_train, X_valid, y_valid)
        print(f"\n[{pos}] FFNN metrics on TEST:")
        evaluate_model(model_ffnn, X_test, y_test, X_train, y_train, X_valid, y_valid)

        model_cat = train_catboost(X_train, y_train, X_valid, y_valid)
        print(f"\n[{pos}] CatBoost metrics on TEST:")
        evaluate_model(model_cat, X_test, y_test, X_train, y_train, X_valid, y_valid)

        # Optional: learning curves for CatBoost per position
        plot_learning_curves(model_cat)

        # ---- Seen vs Cold-start within this position ----
        if train_names is not None and test_names is not None:
            seen_players = set(train_names.unique())
            test_seen_mask = test_names.isin(seen_players)
            test_cold_mask = ~test_seen_mask

            print(f"\n[{pos}] Test composition:")
            print(f"  Seen player rows:      {int(test_seen_mask.sum())}")
            print(f"  Cold-start player rows: {int(test_cold_mask.sum())}")

            def eval_subset(model, X_te, y_te, mask, label):
                n = int(mask.sum())
                if n == 0:
                    print(f"{label}: no rows.")
                    return
                metrics = evaluate_model(model, X_te[mask], y_te[mask])
                print(f"{label}: n={n} | {metrics}")

            print(f"\n[{pos}] CatBoost — Seen vs Cold-start:")
            eval_subset(model_cat, X_test, y_test, test_seen_mask, "TEST (seen players)")
            eval_subset(model_cat, X_test, y_test, test_cold_mask, "TEST (cold-start players)")

            print(f"\n[{pos}] FFNN — Seen vs Cold-start:")
            eval_subset(model_ffnn, X_test, y_test, test_seen_mask, "TEST (seen players)")
            eval_subset(model_ffnn, X_test, y_test, test_cold_mask, "TEST (cold-start players)")

        # Keep models if you want to save/use later
        all_models[pos] = {'ffnn': model_ffnn, 'catboost': model_cat}


if __name__ == "__main__":
    main()