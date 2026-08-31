"""
Model Training: LightGBM Regressor
Supports SynapseML / native LightGBM with log_price target training, evaluates performance on true price scale, and persists artifacts.
Supports PySpark cluster execution with native LightGBM engine fallback.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import calculate_metrics, create_spark_session, get_project_root, save_json

logger = logging.getLogger("uk-housing-train-lgb")


def run_lightgbm():
    start_time = time.time()
    logger.info("Starting LightGBM Training Pipeline...")

    root_dir = get_project_root()
    train_path = root_dir / "data" / "gold" / "train_features.parquet"
    test_path = root_dir / "data" / "gold" / "test_features.parquet"
    feature_meta_path = root_dir / "results" / "feature_names.json"
    model_save_path = root_dir / "results" / "models" / "lightgbm.spark"
    predictions_path = root_dir / "results" / "lightgbm_predictions.parquet"
    metrics_path = root_dir / "results" / "lightgbm_metrics.json"

    if not train_path.exists() or not test_path.exists():
        logger.error("Train or test datasets not found. Ensure split_data.py has been executed.")
        sys.exit(1)

    num_leaves = 63
    learning_rate = 0.05
    n_estimators = 300
    seed = 42

    spark = None
    synapse_available = False
    integration_mode = "LightGBM Native (OpenMP / Parallel)"

    try:
        spark = create_spark_session(app_name="uk-housing-train-lgb")
        from pyspark.sql import functions as F
        from synapse.ml.lightgbm import LightGBMRegressor

        train_df = spark.read.parquet(str(train_path))
        test_df = spark.read.parquet(str(test_path))

        lgb_estimator = LightGBMRegressor(
            featuresCol="features",
            labelCol="log_price",
            predictionCol="raw_prediction",
            numLeaves=num_leaves,
            learningRate=learning_rate,
            numIterations=n_estimators,
            seed=seed,
        )

        logger.info("Fitting SynapseML LightGBM model...")
        t0 = time.perf_counter()
        model = lgb_estimator.fit(train_df)
        training_time_sec = round(time.perf_counter() - t0, 2)

        t1 = time.perf_counter()
        predictions = model.transform(test_df)
        predictions = predictions.withColumn("prediction", F.exp(F.col("raw_prediction")))
        pred_count = predictions.count()
        prediction_time_sec = round(time.perf_counter() - t1, 2)

        metrics = calculate_metrics(predictions, label_col="price", prediction_col="prediction")

        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.write().overwrite().save(str(model_save_path))
        predictions.select("price", "prediction").write.mode("overwrite").parquet(str(predictions_path))
        synapse_available = True
        integration_mode = "SynapseML"

    except Exception as e:
        logger.warning(f"SynapseML Spark unavailable ({e}). Executing via native LightGBM engine...")
        import lightgbm as lgb

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

        logger.info(f"Training LightGBM model on {len(X_train):,} samples...")
        t0 = time.perf_counter()
        model = lgb.LGBMRegressor(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        training_time_sec = round(max(time.perf_counter() - t0, 0.001), 4)
        logger.info(f"LightGBM training completed in {training_time_sec:.4f} seconds.")

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
        model_file = model_save_path / "lightgbm_model.txt"
        model.booster_.save_model(str(model_file))

        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        if predictions_path.is_dir():
            import shutil
            shutil.rmtree(predictions_path)
        test_pdf[["price", "prediction"]].to_parquet(predictions_path, index=False)

    finally:
        if spark:
            spark.stop()

    result_payload = {
        "model": "LightGBM",
        "training_time_sec": training_time_sec,
        "prediction_time_sec": prediction_time_sec,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "integration": integration_mode,
        "hyperparameters": {
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "seed": seed,
        },
    }
    save_json(result_payload, metrics_path)

    total_elapsed = time.time() - start_time
    logger.info(f"LightGBM pipeline finished in {total_elapsed:.2f}s | Results: {result_payload}")


if __name__ == "__main__":
    run_lightgbm()
