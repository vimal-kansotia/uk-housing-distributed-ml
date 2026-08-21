"""
Model Training: LightGBM Regressor
Supports SynapseML LightGBMRegressor with high-performance PySpark / LightGBM fallback,
evaluates regression performance, and persists model artifacts.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StructField, StructType

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

    spark = create_spark_session(app_name="uk-housing-train-lgb")

    # Hyperparameters
    num_leaves = 31
    learning_rate = 0.1
    n_estimators = 100
    seed = 42

    try:
        logger.info(f"Loading train dataset from: {train_path}")
        train_df = spark.read.parquet(str(train_path))

        logger.info(f"Loading test dataset from: {test_path}")
        test_df = spark.read.parquet(str(test_path))

        # Check feature metadata
        if feature_meta_path.exists():
            with open(feature_meta_path, "r") as f:
                feature_meta = json.load(f)
            feature_cols = feature_meta.get("feature_columns", [
                "year", "month", "quarter", "property_type_idx",
                "new_build_idx", "duration_idx", "county_idx",
                "district_idx", "town_idx"
            ])
        else:
            feature_cols = [
                "year", "month", "quarter", "property_type_idx",
                "new_build_idx", "duration_idx", "county_idx",
                "district_idx", "town_idx"
            ]

        # 1. Attempt SynapseML LightGBM integration
        synapse_available = False
        lgb_estimator = None
        integration_mode = "SynapseML"

        try:
            from synapse.ml.lightgbm import LightGBMRegressor
            logger.info("SynapseML LightGBM is available. Using distributed SynapseML LightGBMRegressor.")
            lgb_estimator = LightGBMRegressor(
                featuresCol="features",
                labelCol="price",
                numLeaves=num_leaves,
                learningRate=learning_rate,
                numIterations=n_estimators,
                seed=seed,
            )
            synapse_available = True
        except ImportError:
            logger.warning(
                "SynapseML package (synapse.ml.lightgbm) is not installed in the Spark runtime.\n"
                "To enable native SynapseML, install via: com.microsoft.azure:synapseml_2.12:1.0.4.\n"
                "Proceeding with high-efficiency LightGBM distributed/native pipeline."
            )
            integration_mode = "LightGBM Native (Spark-Integrated)"

        if synapse_available:
            # Fit using SynapseML
            logger.info("Fitting SynapseML LightGBM model...")
            t0 = time.perf_counter()
            model = lgb_estimator.fit(train_df)
            training_time_sec = round(time.perf_counter() - t0, 2)

            t1 = time.perf_counter()
            predictions = model.transform(test_df)
            pred_count = predictions.count()
            prediction_time_sec = round(time.perf_counter() - t1, 2)

            metrics = calculate_metrics(predictions, label_col="price", prediction_col="prediction")

            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            model.write().overwrite().save(str(model_save_path))
            predictions.select("price", "prediction").write.mode("overwrite").parquet(str(predictions_path))

        else:
            import lightgbm as lgb

            logger.info(f"Selecting features for training: {feature_cols}")
            total_train_rows = train_df.count()
            max_sample_rows = 500_000

            if total_train_rows > max_sample_rows:
                sample_fraction = max_sample_rows / total_train_rows
                logger.info(f"Sampling training set from {total_train_rows:,} to ~{max_sample_rows:,} records (fraction={sample_fraction:.4f}) for safe in-memory boosting...")
                train_subset = train_df.sample(withReplacement=False, fraction=sample_fraction, seed=seed)
            else:
                train_subset = train_df

            train_pdf = train_subset.select(feature_cols + ["price"]).toPandas()
            X_train = train_pdf[feature_cols].values
            y_train = train_pdf["price"].values

            logger.info(f"Training LightGBM model on {len(X_train):,} samples...")
            t0 = time.perf_counter()
            model = lgb.LGBMRegressor(
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            training_time_sec = round(time.perf_counter() - t0, 2)
            logger.info(f"LightGBM training completed in {training_time_sec:.2f} seconds.")

            # Test inference via Pandas / PyArrow
            logger.info("Generating predictions on test set...")
            total_test_rows = test_df.count()
            if total_test_rows > 200_000:
                sample_test_fraction = 200_000 / total_test_rows
                logger.info(f"Evaluating test set on ~200,000 test sample (fraction={sample_test_fraction:.4f})...")
                test_subset = test_df.sample(withReplacement=False, fraction=sample_test_fraction, seed=seed)
            else:
                test_subset = test_df

            test_pdf = test_subset.select(feature_cols + ["price"]).toPandas()
            X_test = test_pdf[feature_cols].values

            t1 = time.perf_counter()
            preds = model.predict(X_test)
            prediction_time_sec = round(time.perf_counter() - t1, 2)

            test_pdf["prediction"] = preds
            pred_spark_df = spark.createDataFrame(test_pdf[["price", "prediction"]])
            metrics = calculate_metrics(pred_spark_df, label_col="price", prediction_col="prediction")

            # Save model
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            model_save_path.mkdir(parents=True, exist_ok=True)
            model_file = model_save_path / "lightgbm_model.txt"
            model.booster_.save_model(str(model_file))
            logger.info(f"Saved LightGBM model to: {model_file}")

            # Save predictions
            logger.info(f"Saving predictions to: {predictions_path}")
            pred_spark_df.write.mode("overwrite").parquet(str(predictions_path))

        # Save metrics payload
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

    except Exception as e:
        logger.error(f"Error during LightGBM training: {str(e)}", exc_info=True)
        spark.stop()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    run_lightgbm()
