"""
Shared utility functions for the UK Housing Price Prediction pipeline.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# Configure standard logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("uk-housing-ml")


def get_project_root() -> Path:
    """
    Returns the project root directory.
    Handles running inside Docker container (/opt/spark/work-dir)
    or directly on host system.
    """
    if os.path.exists("/opt/spark/work-dir"):
        return Path("/opt/spark/work-dir")
    # If running locally from repository
    return Path(__file__).resolve().parent.parent


import shutil

def is_spark_available() -> bool:
    """Check if Java is available to run PySpark."""
    return shutil.which("java") is not None


def get_spark_master() -> str:
    """
    Determine the Spark Master URL.
    Defaults to spark://spark-master:7077 if in Docker cluster,
    or local[*] if running on local host.
    """
    if "SPARK_MASTER" in os.environ:
        return os.environ["SPARK_MASTER"]
    if os.path.exists("/opt/spark/work-dir"):
        return "spark://spark-master:7077"
    return "local[*]"


def create_spark_session(
    app_name: str = "uk-housing-ml",
    master: Optional[str] = None,
    extra_configs: Optional[Dict[str, str]] = None,
) -> SparkSession:
    """
    Initializes and returns a configured SparkSession.
    """
    if not is_spark_available():
        raise RuntimeError("Java runtime not found on host. Falling back to native/Arrow execution.")

    if master is None:
        master = get_spark_master()

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.default.parallelism", "16")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.memory.fraction", "0.7")
        .config("spark.memory.storageFraction", "0.3")
    )

    if extra_configs:
        for k, v in extra_configs.items():
            builder = builder.config(k, v)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"SparkSession created: appName='{app_name}', master='{master}', Spark version='{spark.version}'")
    return spark


def log_dataframe_info(df: DataFrame, label: str = "DataFrame") -> Dict[str, Any]:
    """
    Logs metadata about a DataFrame (schema, row count, null count)
    using efficient Spark aggregations without collecting large data to driver.
    Safely skips VectorUDT and complex types for isnan checks.
    """
    row_count = df.count()
    col_count = len(df.columns)
    logger.info(f"=== {label} Summary ===")
    logger.info(f"Total Rows: {row_count:,} | Total Columns: {col_count}")
    logger.info(f"Schema:\n" + "\n".join([f"  - {field.name} ({field.dataType.simpleString()})" for field in df.schema]))

    # Efficient null count per column using single aggregation, safely handling Vector/Complex types
    null_exprs = []
    for c, t in df.dtypes:
        dtype_lower = str(t).lower()
        if "vector" in dtype_lower or "struct" in dtype_lower or "array" in dtype_lower or "map" in dtype_lower:
            null_exprs.append(F.count(F.when(F.col(c).isNull(), c)).alias(c))
        elif "double" in dtype_lower or "float" in dtype_lower:
            null_exprs.append(F.count(F.when(F.col(c).isNull() | F.isnan(F.col(c)), c)).alias(c))
        else:
            null_exprs.append(F.count(F.when(F.col(c).isNull(), c)).alias(c))

    null_counts_row = df.select(null_exprs).first()
    null_summary = null_counts_row.asDict() if null_counts_row else {}
    logger.info(f"Null Counts per Column: {null_summary}")

    return {
        "label": label,
        "row_count": row_count,
        "col_count": col_count,
        "columns": df.columns,
        "null_counts": null_summary,
    }


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and numeric types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: Any, filepath: os.PathLike) -> None:
    """
    Saves a dictionary or list to JSON file, creating parent directories if needed.
    """
    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    logger.info(f"Saved JSON data to {target_path}")


def calculate_metrics(
    predictions_df: DataFrame,
    label_col: str = "price",
    prediction_col: str = "prediction",
) -> Dict[str, float]:
    """
    Evaluates regression predictions using Spark RegressionEvaluator.
    Returns dictionary with MAE, RMSE, and R² rounded to 4 decimal places.
    """
    evaluator_mae = RegressionEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="mae"
    )
    evaluator_rmse = RegressionEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="rmse"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol=label_col, predictionCol=prediction_col, metricName="r2"
    )

    # Filter out any potential NaN/Null in predictions or target before evaluation
    clean_preds = predictions_df.filter(
        F.col(prediction_col).isNotNull() & ~F.isnan(F.col(prediction_col))
    )

    mae = float(evaluator_mae.evaluate(clean_preds))
    rmse = float(evaluator_rmse.evaluate(clean_preds))
    r2 = float(evaluator_r2.evaluate(clean_preds))

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }
    logger.info(f"Calculated Metrics on {clean_preds.count():,} rows -> MAE: {mae:,.2f}, RMSE: {rmse:,.2f}, R²: {r2:.4f}")
    return metrics


def load_json(filepath: os.PathLike) -> Any:
    """
    Loads JSON file from disk.
    """
    target_path = Path(filepath)
    if not target_path.exists():
        raise FileNotFoundError(f"JSON file not found at {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)

