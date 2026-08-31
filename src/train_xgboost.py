"""
Model Training: Distributed / Native XGBoost Regressor
Trains an optimized XGBoost regressor on log_price target, evaluates performance on true price scale, and persists artifacts.
Supports PySpark cluster execution with native XGBoost engine fallback.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import calculate_metrics, create_spark_session, get_project_root, save_json

logger = logging.getLogger("uk-housing-train-xgb")


def run_xgboost():
    start_time = time.time()
    logger.info("Starting XGBoost Training Pipeline...")

    root_dir = get_project_root()
    train_path = root_dir / "data" / "gold" / "train_features.parquet"
    test_path = root_dir / "data" / "gold" / "test_features.parquet"
    feature_meta_path = root_dir / "results" / "feature_names.json"
    model_save_path = root_dir / "results" / "models" / "xgboost.spark"
    predictions_path = root_dir / "results" / "xgboost_predictions.parquet"
    metrics_path = root_dir / "results" / "xgboost_metrics.json"

    if not train_path.exists() or not test_path.exists():
        logger.error(f"Train or test datasets not found. Ensure split_data.py has been executed.")
        sys.exit(1)

    max_depth = 6
    learning_rate = 0.05
    n_estimators = 250
    seed = 42

    spark = None
    spark_xgb_available = False
    integration_mode = "XGBoost Native (Python / OpenMP)"

    try:
        spark = create_spark_session(app_name="uk-housing-train-xgb")
        from pyspark.sql import functions as F
        from xgboost.spark import SparkXGBRegressor

        train_df = spark.read.parquet(str(train_path))
        test_df = spark.read.parquet(str(test_path))

        xgb_estimator = SparkXGBRegressor(
            features_col="features",
            label_col="log_price",
            prediction_col="raw_prediction",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=seed,
        )

        logger.info("Fitting SparkXGBRegressor on cluster...")
        t0 = time.perf_counter()
        xgb_model = xgb_estimator.fit(train_df)
        training_time_sec = round(time.perf_counter() - t0, 2)

        t1 = time.perf_counter()
        predictions = xgb_model.transform(test_df)
        predictions = predictions.withColumn("prediction", F.exp(F.col("raw_prediction")))
        pred_count = predictions.count()
        prediction_time_sec = round(time.perf_counter() - t1, 2)

        metrics = calculate_metrics(predictions, label_col="price", prediction_col="prediction")

        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        xgb_model.save(str(model_save_path))
        predictions.select("price", "prediction").write.mode("overwrite").parquet(str(predictions_path))
        spark_xgb_available = True
        integration_mode = "xgboost.spark"

    except Exception as e:
        logger.warning(f"Spark XGBoost unavailable ({e}). Executing via native XGBoost engine...")
        import xgboost as xgb

        train_pdf = pd.read_parquet(train_path)
        test_pdf = pd.read_parquet(test_path)

        if feature_meta_path.exists():
            with open(feature_meta_path, "r") as f:
                feature_meta = json.load(f)
            feature_cols = feature_meta.get("feature_columns", [c for c in train_pdf.columns if c not in ["price", "log_price", "features", "town", "district", "county", "property_type", "duration", "new_build"]])
        else:
            feature_cols = [c for c in train_pdf.columns if c not in ["price", "log_price", "features", "town", "district", "county", "property_type", "duration", "new_build"]]

        X_train = train_pdf[feature_cols].values
        y_train = train_pdf["log_price"].values
        X_test = test_pdf[feature_cols].values

        t0 = time.perf_counter()
        model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            subsample=0.85,
            colsample_bytree=0.85,
        )
        model.fit(X_train, y_train)
        training_time_sec = round(max(time.perf_counter() - t0, 0.001), 4)
        logger.info(f"XGBoost training completed in {training_time_sec:.4f}s.")

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
        model_save_path.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_save_path / "xgboost_model.json"))

        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        if predictions_path.is_dir():
            import shutil
            shutil.rmtree(predictions_path)
        test_pdf[["price", "prediction"]].to_parquet(predictions_path, index=False)

    finally:
        if spark:
            spark.stop()

    result_payload = {
        "model": "XGBoost",
        "training_time_sec": training_time_sec,
        "prediction_time_sec": prediction_time_sec,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "integration": integration_mode,
        "hyperparameters": {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "seed": seed,
        },
    }
    save_json(result_payload, metrics_path)

    total_elapsed = time.time() - start_time
    logger.info(f"XGBoost pipeline finished in {total_elapsed:.2f}s | Results: {result_payload}")


if __name__ == "__main__":
    run_xgboost()
