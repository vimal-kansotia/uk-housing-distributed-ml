"""
Feature Engineering Phase: Silver -> Gold Parquet Layer
Extracts temporal inflation index, computes smoothed target encodings for high-cardinality geography,
encodes interactions, generates log_price target, and assembles ML feature vectors.
Supports PySpark cluster execution with PyArrow/Pandas engine fallback.
"""

import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils import create_spark_session, get_project_root, log_dataframe_info, save_json

logger = logging.getLogger("uk-housing-features")


def run_feature_engineering():
    start_time = time.time()
    logger.info("Starting Silver -> Gold Feature Engineering Phase (Advanced Encoding & Log Target)...")

    root_dir = get_project_root()
    silver_path = root_dir / "data" / "silver"
    gold_path = root_dir / "data" / "gold" / "gold_features.parquet"
    feature_names_path = root_dir / "results" / "feature_names.json"

    if not silver_path.exists():
        logger.error(f"Silver Parquet directory not found at: {silver_path}")
        sys.exit(1)

    spark = None
    try:
        spark = create_spark_session(app_name="uk-housing-features")
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import StringIndexer, VectorAssembler
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType

        logger.info(f"Loading Silver Parquet from {silver_path} via Spark...")
        df = spark.read.parquet(str(silver_path))

        # Temporal Feature Extraction
        df = (
            df.withColumn("year", F.year(F.col("date")).cast("integer"))
            .withColumn("month", F.month(F.col("date")).cast("integer"))
            .withColumn("quarter", F.quarter(F.col("date")).cast("integer"))
            .withColumn("time_index", ((F.col("year") - 1995) * 12 + F.col("month")).cast("double"))
        )

        df = df.withColumn("log_price", F.log(F.col("price")).cast(DoubleType()))

        df = (
            df.withColumn("prop_x_dur", F.concat_ws("_", F.col("property_type"), F.col("duration")))
            .withColumn("town_x_prop", F.concat_ws("_", F.col("town"), F.col("property_type")))
        )

        global_mean_row = df.select(F.mean("log_price").alias("mean_log")).first()
        global_mean = float(global_mean_row["mean_log"]) if global_mean_row else 12.5
        smoothing = 10.0

        target_enc_cols = ["town", "district", "county", "property_type", "duration", "new_build", "prop_x_dur", "town_x_prop"]
        encoded_df = df
        te_feature_names = []

        for col_name in target_enc_cols:
            te_col_name = f"{col_name}_te"
            stats_df = (
                df.groupBy(col_name)
                .agg(
                    F.count("log_price").alias(f"_cnt_{col_name}"),
                    F.mean("log_price").alias(f"_mean_{col_name}")
                )
            )
            stats_df = stats_df.withColumn(
                te_col_name,
                (
                    (F.col(f"_cnt_{col_name}") * F.col(f"_mean_{col_name}") + F.lit(smoothing * global_mean))
                    / (F.col(f"_cnt_{col_name}") + F.lit(smoothing))
                ).cast(DoubleType())
            ).select(col_name, te_col_name)

            encoded_df = encoded_df.join(stats_df, on=col_name, how="left")
            encoded_df = encoded_df.na.fill({te_col_name: global_mean})
            te_feature_names.append(te_col_name)

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
        pipeline_model = pipeline.fit(encoded_df)
        indexed_df = pipeline_model.transform(encoded_df)

        for _, idx_col in cat_columns:
            indexed_df = indexed_df.withColumn(idx_col, F.col(idx_col).cast("double"))

        feature_cols = [
            "year",
            "month",
            "quarter",
            "time_index",
            "property_type_idx",
            "new_build_idx",
            "duration_idx",
            "county_idx",
            "district_idx",
            "town_idx",
        ] + te_feature_names

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip",
        )
        gold_df = assembler.transform(indexed_df)

        selected_cols = feature_cols + ["features", "price", "log_price", "town", "district", "county", "property_type", "duration", "new_build"]
        final_gold_df = gold_df.select(selected_cols).dropna()

        final_count = final_gold_df.count()
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        final_gold_df.write.mode("overwrite").parquet(str(gold_path))

    except Exception as e:
        logger.warning(f"Spark unavailable ({e}). Executing Feature Engineering via PyArrow / Pandas engine...")
        pdf = pd.read_parquet(silver_path)
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf["year"] = pdf["date"].dt.year.astype(int)
        pdf["month"] = pdf["date"].dt.month.astype(int)
        pdf["quarter"] = pdf["date"].dt.quarter.astype(int)
        pdf["time_index"] = ((pdf["year"] - 1995) * 12 + pdf["month"]).astype(float)
        pdf["log_price"] = np.log(pdf["price"]).astype(float)

        pdf["prop_x_dur"] = pdf["property_type"].astype(str) + "_" + pdf["duration"].astype(str)
        pdf["town_x_prop"] = pdf["town"].astype(str) + "_" + pdf["property_type"].astype(str)

        global_mean = float(pdf["log_price"].mean())
        smoothing = 10.0
        target_enc_cols = ["town", "district", "county", "property_type", "duration", "new_build", "prop_x_dur", "town_x_prop"]
        te_feature_names = []

        for col in target_enc_cols:
            te_name = f"{col}_te"
            counts = pdf[col].value_counts()
            means = pdf.groupby(col, observed=False)["log_price"].mean()
            smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)
            pdf[te_name] = pdf[col].map(smoothed).astype(float).fillna(global_mean)
            te_feature_names.append(te_name)

        cat_columns = [
            ("property_type", "property_type_idx"),
            ("new_build", "new_build_idx"),
            ("duration", "duration_idx"),
            ("county", "county_idx"),
            ("district", "district_idx"),
            ("town", "town_idx"),
        ]

        for orig, idx in cat_columns:
            counts = pdf[orig].value_counts()
            cat_map = {val: i for i, val in enumerate(counts.index)}
            pdf[idx] = pdf[orig].map(cat_map).astype(float).fillna(len(cat_map))

        feature_cols = [
            "year",
            "month",
            "quarter",
            "time_index",
            "property_type_idx",
            "new_build_idx",
            "duration_idx",
            "county_idx",
            "district_idx",
            "town_idx",
        ] + te_feature_names

        final_count = len(pdf)
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        if gold_path.is_dir():
            shutil.rmtree(gold_path)
        pdf.to_parquet(gold_path, index=False)

    finally:
        if spark:
            spark.stop()

    feature_metadata = {
        "feature_columns": feature_cols,
        "target_column": "log_price",
        "raw_target_column": "price",
        "vector_column": "features",
        "total_features": len(feature_cols),
        "categorical_indexed": [idx for _, idx in cat_columns],
        "target_encoded_features": te_feature_names,
        "temporal_features": ["year", "month", "quarter", "time_index"],
        "total_records": final_count,
    }
    save_json(feature_metadata, feature_names_path)

    elapsed_sec = time.time() - start_time
    logger.info(f"Feature Engineering completed in {elapsed_sec:.2f} seconds. Features: {len(feature_cols)}")


if __name__ == "__main__":
    run_feature_engineering()
