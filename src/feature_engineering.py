"""
Feature Engineering Phase: Silver -> Gold Parquet Layer
Extracts temporal features, encodes categoricals using StringIndexer,
assembles ML feature vector, and saves gold features dataset and metadata.
"""

import logging
import sys
import time
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import functions as F

from utils import create_spark_session, get_project_root, log_dataframe_info, save_json

logger = logging.getLogger("uk-housing-features")


def run_feature_engineering():
    start_time = time.time()
    logger.info("Starting Silver -> Gold Feature Engineering Phase...")

    root_dir = get_project_root()
    silver_path = root_dir / "data" / "silver"
    gold_path = root_dir / "data" / "gold" / "gold_features.parquet"
    feature_names_path = root_dir / "results" / "feature_names.json"

    if not silver_path.exists():
        logger.error(f"Silver Parquet directory not found at: {silver_path}")
        sys.exit(1)

    spark = create_spark_session(app_name="uk-housing-features")

    try:
        logger.info(f"Loading Silver Parquet from {silver_path}")
        df = spark.read.parquet(str(silver_path))

        # 1. Temporal Feature Extraction
        logger.info("Extracting temporal features (year, month, quarter) from date column...")
        df = (
            df.withColumn("year", F.year(F.col("date")).cast("integer"))
            .withColumn("month", F.month(F.col("date")).cast("integer"))
            .withColumn("quarter", F.quarter(F.col("date")).cast("integer"))
        )

        # 2. Categorical Encoders using StringIndexer (Pipeline)
        # Low and high-cardinality columns: property_type, new_build, duration, county, district, town
        cat_columns = [
            ("property_type", "property_type_idx"),
            ("new_build", "new_build_idx"),
            ("duration", "duration_idx"),
            ("county", "county_idx"),
            ("district", "district_idx"),
            ("town", "town_idx"),
        ]

        indexers = [
            StringIndexer(inputCol=orig, outputCol=idx, handleInvalid="keep", stringOrderType="frequencyDesc")
            for orig, idx in cat_columns
        ]

        pipeline = Pipeline(stages=indexers)
        logger.info("Fitting StringIndexer pipeline on dataset...")
        pipeline_model = pipeline.fit(df)
        indexed_df = pipeline_model.transform(df)

        # Cast index columns to double explicitly for MLlib compatibility
        for _, idx_col in cat_columns:
            indexed_df = indexed_df.withColumn(idx_col, F.col(idx_col).cast("double"))

        feature_cols = [
            "year",
            "month",
            "quarter",
            "property_type_idx",
            "new_build_idx",
            "duration_idx",
            "county_idx",
            "district_idx",
            "town_idx",
        ]

        # VectorAssembler for Spark ML models
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip",
        )
        gold_df = assembler.transform(indexed_df)

        # Target and feature columns to keep
        selected_cols = feature_cols + ["features", "price"]
        final_gold_df = gold_df.select(selected_cols).dropna()

        final_count = final_gold_df.count()
        logger.info(f"Gold Layer Records: {final_count:,} with {len(feature_cols)} engineered features.")

        log_dataframe_info(final_gold_df, label="Gold Layer Features")

        logger.info("Displaying 5 sample rows from Gold dataset:")
        final_gold_df.select(feature_cols + ["price"]).show(5)

        # Save gold features Parquet
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing Gold Parquet to: {gold_path}")
        final_gold_df.write.mode("overwrite").parquet(str(gold_path))

        # Save feature names and metadata JSON
        feature_metadata = {
            "feature_columns": feature_cols,
            "target_column": "price",
            "vector_column": "features",
            "total_features": len(feature_cols),
            "categorical_indexed": [idx for _, idx in cat_columns],
            "temporal_features": ["year", "month", "quarter"],
            "total_records": final_count,
        }
        save_json(feature_metadata, feature_names_path)

        elapsed_sec = time.time() - start_time
        logger.info(f"Feature Engineering completed in {elapsed_sec:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error during Feature Engineering: {str(e)}", exc_info=True)
        spark.stop()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    run_feature_engineering()
