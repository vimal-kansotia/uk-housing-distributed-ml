"""
Aggregate Results: Aggregates metrics from all four trained models into a single CSV and summary table.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

from utils import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("uk-housing-aggregate")


def aggregate_model_metrics():
    logger.info("Aggregating model evaluation metrics...")
    root_dir = get_project_root()
    results_dir = root_dir / "results"

    metric_files = [
        results_dir / "linear_regression_metrics.json",
        results_dir / "random_forest_metrics.json",
        results_dir / "xgboost_metrics.json",
        results_dir / "lightgbm_metrics.json",
    ]

    records = []
    for m_file in metric_files:
        if not m_file.exists():
            logger.warning(f"Metrics file not found: {m_file}. Skipping.")
            continue

        try:
            with open(m_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            records.append({
                "model": data.get("model", m_file.stem.replace("_metrics", "")),
                "training_time_sec": data.get("training_time_sec", None),
                "prediction_time_sec": data.get("prediction_time_sec", None),
                "mae": data.get("mae", None),
                "rmse": data.get("rmse", None),
                "r2": data.get("r2", None),
            })
            logger.info(f"Loaded metrics from {m_file.name}")
        except Exception as e:
            logger.error(f"Failed to read {m_file}: {e}")

    if not records:
        logger.error("No metrics records found to aggregate.")
        sys.exit(1)

    df_comparison = pd.DataFrame(records)

    # Ensure required columns order
    cols_order = ["model", "training_time_sec", "prediction_time_sec", "mae", "rmse", "r2"]
    df_comparison = df_comparison[cols_order]

    output_csv = results_dir / "model_comparison.csv"
    df_comparison.to_csv(output_csv, index=False)
    logger.info(f"Model comparison saved to: {output_csv}")

    print("\n" + "=" * 80)
    print("                      UK HOUSING PRICE PREDICTION — MODEL BENCHMARK")
    print("=" * 80)
    print(df_comparison.to_string(index=False))
    print("=" * 80 + "\n")

    # Determine best models
    if "rmse" in df_comparison and not df_comparison["rmse"].dropna().empty:
        best_rmse = df_comparison.loc[df_comparison["rmse"].idxmin()]
        logger.info(f"Best Model by RMSE: {best_rmse['model']} (RMSE: {best_rmse['rmse']:,.2f})")

    if "r2" in df_comparison and not df_comparison["r2"].dropna().empty:
        best_r2 = df_comparison.loc[df_comparison["r2"].idxmax()]
        logger.info(f"Best Model by R²:   {best_r2['model']} (R²: {best_r2['r2']:.4f})")

    if "training_time_sec" in df_comparison and not df_comparison["training_time_sec"].dropna().empty:
        fastest_train = df_comparison.loc[df_comparison["training_time_sec"].idxmin()]
        logger.info(f"Fastest Training:   {fastest_train['model']} ({fastest_train['training_time_sec']:.2f}s)")


if __name__ == "__main__":
    aggregate_model_metrics()
