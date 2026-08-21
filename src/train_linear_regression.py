"""
Model Training: Linear Regression Baseline
Trains a Spark MLlib LinearRegression model, evaluates performance, and persists artifacts.
"""

import logging
import sys
import time
from pathlib import Path

from pyspark.ml.regression import LinearRegression

from utils import calculate_metrics, create_spark_session, get_project_root, save_json

logger = logging.getLogger("uk-housing-train-lr")


def run_linear_regression():
    start_time = time.time()
    logger.info("Starting Linear Regression Training Pipeline...")

    root_dir = get_project_root()
    train_path = root_dir / "data" / "gold" / "train_features.parquet"
    test_path = root_dir / "data" / "gold" / "test_features.parquet"
    model_save_path = root_dir / "results" / "models" / "linear_regression.spark"
    predictions_path = root_dir / "results" / "linear_regression_predictions.parquet"
    metrics_path = root_dir / "results" / "linear_regression_metrics.json"

    if not train_path.exists() or not test_path.exists():
        logger.error(f"Train or test datasets not found. Ensure split_data.py has been executed.")
        sys.exit(1)

    spark = create_spark_session(app_name="uk-housing-train-lr")

    try:
        logger.info(f"Loading train dataset from: {train_path}")
        train_df = spark.read.parquet(str(train_path))

        logger.info(f"Loading test dataset from: {test_path}")
        test_df = spark.read.parquet(str(test_path))

        # Initialize LinearRegression
        lr = LinearRegression(
            featuresCol="features",
            labelCol="price",
            maxIter=50,
            regParam=0.1,
            elasticNetParam=0.0,
        )

        # Measure training time
        logger.info("Fitting Linear Regression model on training data...")
        t0 = time.perf_counter()
        lr_model = lr.fit(train_df)
        training_time_sec = round(time.perf_counter() - t0, 2)
        logger.info(f"Linear Regression training completed in {training_time_sec:.2f} seconds.")

        # Measure prediction time
        logger.info("Generating predictions on test data...")
        t1 = time.perf_counter()
        predictions = lr_model.transform(test_df)
        # Force evaluation/action to measure real prediction duration
        pred_count = predictions.count()
        prediction_time_sec = round(time.perf_counter() - t1, 2)
        logger.info(f"Generated predictions for {pred_count:,} test records in {prediction_time_sec:.2f} seconds.")

        # Calculate metrics
        logger.info("Evaluating model metrics on test predictions...")
        metrics = calculate_metrics(predictions, label_col="price", prediction_col="prediction")

        # Save model
        logger.info(f"Saving trained model to: {model_save_path}")
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        lr_model.write().overwrite().save(str(model_save_path))

        # Save predictions Parquet
        logger.info(f"Saving predictions to: {predictions_path}")
        predictions.select("price", "prediction").write.mode("overwrite").parquet(str(predictions_path))

        # Prepare metrics payload
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

    except Exception as e:
        logger.error(f"Error during Linear Regression training: {str(e)}", exc_info=True)
        spark.stop()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    run_linear_regression()
