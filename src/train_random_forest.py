"""
Model Training: Random Forest Regressor
Trains a Spark MLlib RandomForestRegressor model, evaluates performance, and persists artifacts.
"""

import logging
import sys
import time
from pathlib import Path

from pyspark.ml.regression import RandomForestRegressor

from utils import calculate_metrics, create_spark_session, get_project_root, save_json

logger = logging.getLogger("uk-housing-train-rf")


def run_random_forest():
    start_time = time.time()
    logger.info("Starting Random Forest Training Pipeline...")

    root_dir = get_project_root()
    train_path = root_dir / "data" / "gold" / "train_features.parquet"
    test_path = root_dir / "data" / "gold" / "test_features.parquet"
    model_save_path = root_dir / "results" / "models" / "random_forest.spark"
    predictions_path = root_dir / "results" / "random_forest_predictions.parquet"
    metrics_path = root_dir / "results" / "random_forest_metrics.json"

    if not train_path.exists() or not test_path.exists():
        logger.error(f"Train or test datasets not found. Ensure split_data.py has been executed.")
        sys.exit(1)

    spark = create_spark_session(app_name="uk-housing-train-rf")

    try:
        logger.info(f"Loading train dataset from: {train_path}")
        train_df = spark.read.parquet(str(train_path))

        logger.info(f"Loading test dataset from: {test_path}")
        test_df = spark.read.parquet(str(test_path))

        # Hyperparameters
        num_trees = 50
        max_depth = 10
        min_instances_per_node = 1
        seed = 42

        logger.info(
            f"Configuring RandomForestRegressor: numTrees={num_trees}, maxDepth={max_depth}, "
            f"minInstancesPerNode={min_instances_per_node}, seed={seed}"
        )

        rf = RandomForestRegressor(
            featuresCol="features",
            labelCol="price",
            numTrees=num_trees,
            maxDepth=max_depth,
            minInstancesPerNode=min_instances_per_node,
            seed=seed,
            maxBins=64,
        )

        # Measure training time
        logger.info("Fitting Random Forest model on distributed Spark cluster...")
        t0 = time.perf_counter()
        rf_model = rf.fit(train_df)
        training_time_sec = round(time.perf_counter() - t0, 2)
        logger.info(f"Random Forest training completed in {training_time_sec:.2f} seconds.")

        # Measure prediction time
        logger.info("Generating predictions on test data...")
        t1 = time.perf_counter()
        predictions = rf_model.transform(test_df)
        pred_count = predictions.count()
        prediction_time_sec = round(time.perf_counter() - t1, 2)
        logger.info(f"Generated predictions for {pred_count:,} test records in {prediction_time_sec:.2f} seconds.")

        # Calculate metrics
        logger.info("Evaluating model metrics on test predictions...")
        metrics = calculate_metrics(predictions, label_col="price", prediction_col="prediction")

        # Save model
        logger.info(f"Saving trained model to: {model_save_path}")
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        rf_model.write().overwrite().save(str(model_save_path))

        # Save predictions Parquet
        logger.info(f"Saving predictions to: {predictions_path}")
        predictions.select("price", "prediction").write.mode("overwrite").parquet(str(predictions_path))

        # Prepare metrics payload
        result_payload = {
            "model": "Random Forest",
            "training_time_sec": training_time_sec,
            "prediction_time_sec": prediction_time_sec,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "hyperparameters": {
                "num_trees": num_trees,
                "max_depth": max_depth,
                "min_instances_per_node": min_instances_per_node,
                "seed": seed,
            },
        }
        save_json(result_payload, metrics_path)

        total_elapsed = time.time() - start_time
        logger.info(f"Random Forest pipeline finished in {total_elapsed:.2f}s | Results: {result_payload}")

    except Exception as e:
        logger.error(f"Error during Random Forest training: {str(e)}", exc_info=True)
        spark.stop()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    run_random_forest()
