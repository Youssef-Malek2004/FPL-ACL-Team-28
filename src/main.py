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

import numpy as np
from typing import Dict, Any
from dataclasses import asdict
import datetime
from scripts.reporting import (now_stamp, plot_parity, plot_residuals, save_json,
                               ensure_dir, permutation_importance, ExperimentReport,
                               write_markdown_report)
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
    p.add_argument(
        "--report_dir",
        default=os.path.join(PROJECT_ROOT, "reports"),
        help="Directory to write experiment reports (default: ./reports)",
    )
    p.add_argument(
        "--run_name",
        default=None,
        help="Optional run name to organize outputs (default: timestamped).",
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

    # print(df_with_lagged_features.columns)
    # print(df_with_lagged_features.columns.value_counts().count())
    # print(df_with_lagged_features.head())

    df_with_target = add_upcoming_total_points(df_with_lagged_features)

    X, y = build_xy(df_with_target, keep_player_id=True, player_col="name_encoded")

    X = X.drop(columns=['total_points'])

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

    # print(f"Seasons by start year (chronological): {years_sorted}")
    # print(f"Train rows: {len(X_train)}, Valid rows: {len(X_valid)}, Test rows: {len(X_test)}")

    # -------- Train final models on (Train, Valid) and evaluate on Test --------
    model_ffnn = train_ffnn(X_train, y_train, X_valid, y_valid)
    rep_ffnn = evaluate_model(model_ffnn, X_test, y_test, X_train, y_train, X_valid, y_valid)

    model_cat = train_catboost(X_train, y_train, X_valid, y_valid)
    rep_cat  = evaluate_model(model_cat,  X_test, y_test, X_train, y_train, X_valid, y_valid)
    # plot_learning_curves(model_cat)

    # -------- Test reporting: Seen vs Cold-start players ----------------------
    seen_players = set(train_names.unique())
    test_seen_mask = test_names.isin(seen_players)
    test_cold_mask = ~test_seen_mask

    print("\n Test composition:")
    print(f"  Seen players rows:      {int(test_seen_mask.sum())}")
    print(f"  Cold-start players rows: {int(test_cold_mask.sum())}")

    def eval_subset_dict(model, X_te, y_te, mask) -> Dict[str, Any]:
        n = int(mask.sum())
        if n == 0:
            return {}
        sub_report = evaluate_model(model, X_te[mask], y_te[mask])
        return {"n": n, "metrics": sub_report.get("test", {})}

    cat_seen  = eval_subset_dict(model_cat,  X_test, y_test, test_seen_mask)
    cat_cold  = eval_subset_dict(model_cat,  X_test, y_test, test_cold_mask)
    ffnn_seen = eval_subset_dict(model_ffnn, X_test, y_test, test_seen_mask)
    ffnn_cold = eval_subset_dict(model_ffnn, X_test, y_test, test_cold_mask)

    # ===== Build report folder =====
    run_id = args.run_name or now_stamp()
    out_dir = ensure_dir(os.path.join(args.report_dir, run_id))
    plots_dir = ensure_dir(os.path.join(out_dir, "plots"))

    # ===== Feature importance (CatBoost) =====
    cat_feat_imps = []
    try:
        # expects same column order as training
        feature_names = list(X_train.columns)
        imps = model_cat.get_feature_importance()
        cat_feat_imps = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)[:40]
    except Exception as e:
        print(f"[warn] CatBoost feature importance failed: {e}")

    # ===== Permutation importance (FFNN) on VALID =====
    ffnn_perm = []
    try:
        r = permutation_importance(
            model_ffnn, X_valid, y_valid, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error"
        )
        # Convert to “importance = -ΔMAE” to keep higher=better intuition
        deltas = -r.importances_mean
        order = np.argsort(deltas)[::-1]
        top = min(40, len(order))
        ffnn_perm = [(X_valid.columns[i], (deltas[i], r.importances_std[i])) for i in order[:top]]
    except Exception as e:
        print(f"[warn] FFNN permutation importance failed: {e}")

    # ===== Plots (Test) =====
    # CatBoost
    yte = y_test.values
    yhat_cat = np.asarray(rep_cat["_predictions"]["test"]).reshape(-1)
    yhat_ff = np.asarray(rep_ffnn["_predictions"]["test"]).reshape(-1)

    cat_residuals_path = os.path.join(plots_dir, "catboost_residuals.png")
    cat_parity_path = os.path.join(plots_dir, "catboost_parity.png")
    ffnn_residuals_path = os.path.join(plots_dir, "ffnn_residuals.png")
    ffnn_parity_path = os.path.join(plots_dir, "ffnn_parity.png")

    plot_residuals(yte, yhat_cat, cat_residuals_path, "CatBoost Residuals (Test)")
    plot_parity(yte, yhat_cat, cat_parity_path, "CatBoost Parity (Test)")
    plot_residuals(yte, yhat_ff, ffnn_residuals_path, "FFNN Residuals (Test)")
    plot_parity(yte, yhat_ff, ffnn_parity_path, "FFNN Parity (Test)")

    # ===== Compose report object =====
    report = ExperimentReport(
        meta={
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "run_name": args.run_name,
            "input_path": input_path,
            "years_sorted": years_sorted,
        },
        data_splits={
            "n_train": len(X_train),
            "n_valid": len(X_valid),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
        },
        models={
            "catboost": {"metrics": {k: v for k, v in rep_cat.items() if k in ("train", "valid", "test")}},
            "ffnn": {"metrics": {k: v for k, v in rep_ffnn.items() if k in ("train", "valid", "test")}},
        },
        subsets={
            "test_seen_rows": int(test_seen_mask.sum()),
            "test_cold_rows": int(test_cold_mask.sum()),
            "catboost": {
                "train": {},  # not computed
                "valid": {},
                "test": {
                    "MAE": rep_cat["test"]["MAE"], "RMSE": rep_cat["test"]["RMSE"], "R2": rep_cat["test"]["R2"]
                },
                "seen": cat_seen.get("metrics", {}),
                "cold": cat_cold.get("metrics", {}),
            },
            "ffnn": {
                "train": {},
                "valid": {},
                "test": {
                    "MAE": rep_ffnn["test"]["MAE"], "RMSE": rep_ffnn["test"]["RMSE"], "R2": rep_ffnn["test"]["R2"]
                },
                "seen": ffnn_seen.get("metrics", {}),
                "cold": ffnn_cold.get("metrics", {}),
            },
        },
        artifacts={
            "catboost_feature_importance": cat_feat_imps,
            "ffnn_permutation_importance": ffnn_perm,
            "plots": {
                "catboost_residuals": os.path.relpath(cat_residuals_path, start=out_dir),
                "catboost_parity": os.path.relpath(cat_parity_path, start=out_dir),
                "ffnn_residuals": os.path.relpath(ffnn_residuals_path, start=out_dir),
                "ffnn_parity": os.path.relpath(ffnn_parity_path, start=out_dir),
            },
        }
    )

    # ===== Write JSON + Markdown =====
    json_path = os.path.join(out_dir, "report.json")
    md_path = os.path.join(out_dir, "report.md")
    save_json(asdict(report), json_path)
    write_markdown_report(report, md_path)

    print(f"\n✓ Reports saved:\n- {md_path}\n- {json_path}\n")

if __name__ == "__main__":
    main()