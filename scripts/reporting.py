# --- Reporting utilities ---
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json, pathlib, datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

@dataclass
class ExperimentReport:
    meta: Dict[str, Any]
    data_splits: Dict[str, Any]
    models: Dict[str, Any]
    subsets: Dict[str, Any]
    artifacts: Dict[str, Any]

def now_stamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_dir(p: str) -> str:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _to_md_metrics_table(title: str, metrics: dict, order=("train", "valid", "test")) -> str:
    lines = [f"### {title}", "", "| Split | MAE | RMSE | R² |", "|---|---:|---:|---:|"]
    any_row = False
    for split in order:
        m = metrics.get(split)
        if not isinstance(m, dict) or "MAE" not in m or "RMSE" not in m or "R2" not in m:
            continue  # skip missing/empty rows
        lines.append(f"| {split} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['R2']:.4f} |")
        any_row = True
    if not any_row:
        lines.append("| _none_ |  |  |  |")
    lines.append("")
    return "\n".join(lines)

def plot_residuals(y_true, y_pred, outpath: str, title: str):
    res = (y_true - y_pred)
    plt.figure()
    plt.hist(res, bins=50)
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_parity(y_true, y_pred, outpath: str, title: str):
    plt.figure()
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def write_markdown_report(report: ExperimentReport, md_path: str):
    m = report.models
    s = report.subsets
    ds = report.data_splits
    art = report.artifacts
    meta = report.meta

    lines = [
        f"# Experiment Report — {meta['run_id']}",
        "",
        f"- **Timestamp:** {meta['timestamp']}",
        f"- **Run Name:** {meta.get('run_name','<none>')}",
        f"- **Input CSV:** `{meta['input_path']}`",
        f"- **Seasons (chronological):** {meta.get('years_sorted', [])}",
        "",
        "## Data & Splits",
        f"- Train rows: {ds['n_train']} | Valid rows: {ds['n_valid']} | Test rows: {ds['n_test']}",
        f"- Features: {ds['n_features']}",
        "",
        _to_md_metrics_table("CatBoost Metrics", m["catboost"]["metrics"]),
        _to_md_metrics_table("FFNN Metrics", m["ffnn"]["metrics"]),
        "## Seen vs Cold-start (Test)",
        "",
        f"- **Test composition** → Seen rows: {s['test_seen_rows']} | Cold-start rows: {s['test_cold_rows']}",
        "",
        _to_md_metrics_table(
            "CatBoost — Seen/Cold on Test",
            {
                "seen": s["catboost"].get("seen", {}),
                "cold": s["catboost"].get("cold", {}),
            },
            order=("seen", "cold"),
        ),

        _to_md_metrics_table(
            "FFNN — Seen/Cold on Test",
            {
                "seen": s["ffnn"].get("seen", {}),
                "cold": s["ffnn"].get("cold", {}),
            },
            order=("seen", "cold"),
        ),
        "## Feature Importances",
        "",
        "- **CatBoost (gain)**:",
        "",
        "| Feature | Importance |",
        "|---|---:|",
    ]
    for feat, val in art.get("catboost_feature_importance", []):
        lines.append(f"| {feat} | {val:.6f} |")
    lines += [
        "",
        "- **FFNN (Permutation Importance on VALID)**:",
        "",
        "| Feature | Mean ΔScore | Std |",
        "|---|---:|---:|",
    ]
    for feat, (mean_imp, std_imp) in art.get("ffnn_permutation_importance", []):
        lines.append(f"| {feat} | {mean_imp:.6f} | {std_imp:.6f} |")
    lines += [
        "",
        "## Plots",
        "",
        f"- Residuals (CatBoost, Test): `{art['plots']['catboost_residuals']}`",
        f"- Parity (CatBoost, Test): `{art['plots']['catboost_parity']}`",
        f"- Residuals (FFNN, Test): `{art['plots']['ffnn_residuals']}`",
        f"- Parity (FFNN, Test): `{art['plots']['ffnn_parity']}`",
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
