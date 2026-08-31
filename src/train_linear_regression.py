"""
Model Training: Linear Regression Baseline
Trains a Linear Regression (Ridge) model on log_price target, evaluates performance on true price scale, and persists artifacts.
Supports PySpark MLlib with scikit-learn fallback.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from utils import calculate_metrics, create_spark_session, get_project_root, save_json

logger = logging.getLogger("uk-housing-train-lr")


def run_linear_regression():
    start_time = time.time()
    logger.info("Starting Linear Regression Training Pipeline...")

    root_dir = get_project_root()
    train_path = root_dir / "data" / "gold" / "train_features.parquet"
    test_path = root_dir / "data" / "gold" / "test_features.parquet"
    feature_meta_path = root_dir / "results" / "feature_names.json"
    model_save_path = root_dir / "results" / "models" / "linear_regression.spark"
    predictions_path = root_dir / "results" / "linear_regression_predictions.parquet"
    metrics_path = root_dir / "results" / "linear_regression_metrics.json"

    if not train_path.exists() or not test_path.exists():
        logger.error(f"Train or test datasets not found. Ensure split_data.py has been executed.")
        sys.exit(1)

    spark = None
    try:
        spark = create_spark_session(app_name="uk-housing-train-lr")
        from pyspark.ml.regression import LinearRegression
        from pyspark.sql import functions as F

        logger.info(f"Loading train dataset from: {train_path} via Spark")
        train_df = spark.read.parquet(str(train_path))
        test_df = spark.read.parquet(str(test_path))

        lr = LinearRegression(
            featuresCol="features",
            labelCol="log_price",
            predictionCol="raw_prediction",
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.0,
        )

        t0 = time.perf_counter()
        lr_model = lr.fit(train_df)
        training_time_sec = round(time.perf_counter() - t0, 2)

        t1 = time.perf_counter()
        predictions = lr_model.transform(test_df)
        predictions = predictions.withColumn("prediction", F.exp(F.col("raw_prediction")))
        pred_count = predictions.count()
        prediction_time_sec = round(time.perf_counter() - t1, 2)

        metrics = calculate_metrics(predictions, label_col="price", prediction_col="prediction")

        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        lr_model.write().overwrite().save(str(model_save_path))
        predictions.select("price", "prediction").write.mode("overwrite").parquet(str(predictions_path))

    except Exception as e:
        logger.warning(f"Spark MLlib unavailable ({e}). Executing via scikit-learn Ridge regression engine...")
        train_pdf = pd.read_parquet(train_path)
        test_pdf = pd.read_parquet(test_path)

        if feature_meta_path.exists():
            with open(feature_meta_path, "r") as f:
                feature_meta = json.load(f)
            feature_cols = feature_meta.get("feature_columns", [c for c in train_pdf.columns if c not in ["price", "log_price", "features", "town", "district", "county", "property_type", "duration", "new_build"]])
        else:
            feature_cols = [c for c in train_pdf.columns if c not in ["price", "log_price", "features", "town", "district", "county", "property_type", "duration", "new_build"]]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_pdf[feature_cols].values)
        y_train = train_pdf["log_price"].values
        X_test = scaler.transform(test_pdf[feature_cols].values)

        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        t0 = time.perf_counter()
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        training_time_sec = round(max(time.perf_counter() - t0, 0.001), 4)

        t1 = time.perf_counter()
        preds_log = model.predict(X_test)
        prediction_time_sec = round(max(time.perf_counter() - t1, 0.001), 4)

        preds_gbp = np.exp(preds_log)
        test_pdf["prediction"] = preds_gbp

        mae = float(mean_absolute_error(test_pdf["price"], preds_gbp))
        rmse = float(np.sqrt(mean_squared_error(test_pdf["price"], preds_gbp)))
        r2 = float(r2_score(test_pdf["price"], preds_gbp))

        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        }
        logger.info(f"Calculated Metrics on {len(test_pdf):,} rows -> MAE: £{mae:,.2f}, RMSE: £{rmse:,.2f}, R²: {r2:.4f}")

        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        if predictions_path.is_dir():
            import shutil
            shutil.rmtree(predictions_path)
        test_pdf[["price", "prediction"]].to_parquet(predictions_path, index=False)

    finally:
        if spark:
            spark.stop()

    result_payload = {
        "model": "Linear Regression",
        "training_time_sec": training_time_sec,
        "prediction_time_sec": prediction_time_sec,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
    }
    save_json(result_payload, metrics_path)

    total_elapsed = time.time() - start_time
    logger.info(f"Linear Regression pipeline finished in {total_elapsed:.2f}s | Results: {result_payload}")


if __name__ == "__main__":
    run_linear_regression()
