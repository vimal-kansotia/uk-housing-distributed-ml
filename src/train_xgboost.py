"""
Model Training: Distributed XGBoost Regressor
Trains a distributed Spark XGBoost regressor, evaluates performance, and persists artifacts.
"""

import logging
import sys
import time
from pathlib import Path

from utils import calculate_metrics, create_spark_session, get_project_root, save_json

logger = logging.getLogger("uk-housing-train-xgb")


def get_xgboost_estimator(features_col="features", label_col="price", max_depth=5, learning_rate=0.1, n_estimators=100, seed=42):
    """
    Attempts to import and instantiate the distributed XGBoost Spark regressor.
    """
    try:
        from pyspark.ml.xgboost import XGBoostRegressor
        logger.info("Using pyspark.ml.xgboost.XGBoostRegressor integration.")
        return XGBoostRegressor(
            featuresCol=features_col,
            labelCol=label_col,
            maxDepth=max_depth,
            learningRate=learning_rate,
            numRound=n_estimators,
            seed=seed,
        ), "pyspark.ml.xgboost"
    except ImportError:
        pass

    try:
        from xgboost.spark import SparkXGBRegressor
        logger.info("Using xgboost.spark.SparkXGBRegressor native distributed integration.")
        return SparkXGBRegressor(
            features_col=features_col,
            label_col=label_col,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=seed,
            num_workers=2,
        ), "xgboost.spark"
    except ImportError as e:
        logger.error(f"Failed to import XGBoost Spark estimator: {e}")
        raise


def run_xgboost():
    start_time = time.time()
    logger.info("Starting Distributed XGBoost Training Pipeline...")

    root_dir = get_project_root()
    train_path = root_dir / "data" / "gold" / "train_features.parquet"
    test_path = root_dir / "data" / "gold" / "test_features.parquet"
    model_save_path = root_dir / "results" / "models" / "xgboost.spark"
    predictions_path = root_dir / "results" / "xgboost_predictions.parquet"
    metrics_path = root_dir / "results" / "xgboost_metrics.json"

    if not train_path.exists() or not test_path.exists():
        logger.error(f"Train or test datasets not found. Ensure split_data.py has been executed.")
        sys.exit(1)

    spark = create_spark_session(app_name="uk-housing-train-xgb")

    try:
        logger.info(f"Loading train dataset from: {train_path}")
        train_df = spark.read.parquet(str(train_path))

        logger.info(f"Loading test dataset from: {test_path}")
        test_df = spark.read.parquet(str(test_path))

        # Hyperparameters
        max_depth = 5
        learning_rate = 0.1
        n_estimators = 100
        seed = 42

        logger.info(
            f"Configuring XGBoost: max_depth={max_depth}, learning_rate={learning_rate}, "
            f"n_estimators={n_estimators}, seed={seed}"
        )

        xgb_estimator, integration_name = get_xgboost_estimator(
            features_col="features",
            label_col="price",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            seed=seed,
        )

        # Measure training time
        logger.info("Fitting XGBoost model on distributed Spark cluster...")
        t0 = time.perf_counter()
        xgb_model = xgb_estimator.fit(train_df)
        training_time_sec = round(time.perf_counter() - t0, 2)
        logger.info(f"XGBoost training completed in {training_time_sec:.2f} seconds.")

        # Measure prediction time
        logger.info("Generating predictions on test data...")
        t1 = time.perf_counter()
        predictions = xgb_model.transform(test_df)
        pred_count = predictions.count()
        prediction_time_sec = round(time.perf_counter() - t1, 2)
        logger.info(f"Generated predictions for {pred_count:,} test records in {prediction_time_sec:.2f} seconds.")

        # Calculate metrics
        logger.info("Evaluating model metrics on test predictions...")
        metrics = calculate_metrics(predictions, label_col="price", prediction_col="prediction")

        # Save model
        logger.info(f"Saving trained model to: {model_save_path}")
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            xgb_model.write().overwrite().save(str(model_save_path))
        except Exception:
            # Fallback to save model if write().overwrite() is structured differently
            xgb_model.save(str(model_save_path))

        # Save predictions Parquet
        logger.info(f"Saving predictions to: {predictions_path}")
        predictions.select("price", "prediction").write.mode("overwrite").parquet(str(predictions_path))

        # Prepare metrics payload
        result_payload = {
            "model": "XGBoost",
            "training_time_sec": training_time_sec,
            "prediction_time_sec": prediction_time_sec,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "integration": integration_name,
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

    except Exception as e:
        logger.error(f"Error during XGBoost training: {str(e)}", exc_info=True)
        spark.stop()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    run_xgboost()
