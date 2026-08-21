"""
Transformation Phase: Bronze -> Silver Parquet Layer
Cleanses, normalizes, validates types, removes duplicates/nulls, and standardizes schema.
Optimized for distributed cluster execution with 32 partitions.
"""

import logging
import sys
import time
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType

from utils import create_spark_session, get_project_root, log_dataframe_info

logger = logging.getLogger("uk-housing-transform")


def run_transformation():
    start_time = time.time()
    logger.info("Starting Bronze -> Silver Transformation Phase...")

    root_dir = get_project_root()
    bronze_path = root_dir / "data" / "bronze"
    silver_path = root_dir / "data" / "silver"
    log_file_path = root_dir / "results" / "transform_log.txt"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    if not bronze_path.exists():
        logger.error(f"Bronze Parquet directory not found at: {bronze_path}")
        sys.exit(1)

    spark = create_spark_session(app_name="uk-housing-transform")
    transform_log = []

    try:
        logger.info(f"Loading Bronze Parquet from {bronze_path} with 32 partitions...")
        df = spark.read.parquet(str(bronze_path)).repartition(32)
        initial_count = df.count()
        transform_log.append(f"Initial Bronze Row Count: {initial_count:,}")
        logger.info(f"Initial Bronze Row Count: {initial_count:,}")

        # Step 1: Standardize Column Names
        logger.info("Step 1: Standardizing Column Names...")
        rename_lookup = {
            "transaction unique identifier": "transaction_id",
            "price": "price",
            "date of transfer": "date",
            "property type": "property_type",
            "old/new": "new_build",
            "duration": "duration",
            "town/city": "town",
            "district": "district",
            "county": "county",
            "ppdcategory type": "ppdcategory_type",
            "record status - monthly file only": "record_status",
        }

        for orig_col in df.columns:
            key = orig_col.strip().lower()
            cleaned_col = rename_lookup.get(key, orig_col.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_"))
            df = df.withColumnRenamed(orig_col, cleaned_col)

        # Step 2: Casting & Standardizing types
        logger.info("Step 2: Casting types and normalizing date/string columns...")
        df = df.withColumn("price", F.col("price").cast(DoubleType()))
        df = df.withColumn(
            "date",
            F.coalesce(
                F.to_date(F.to_timestamp(F.col("date"), "yyyy-MM-dd HH:mm:ss")),
                F.to_date(F.to_timestamp(F.col("date"), "yyyy-MM-dd HH:mm")),
                F.to_date(F.col("date"), "yyyy-MM-dd"),
                F.to_date(F.col("date")),
                F.to_date(F.to_timestamp(F.col("date"))),
            ).cast(DateType()),
        )

        string_cols = ["property_type", "new_build", "duration", "town", "district", "county"]
        for col_name in string_cols:
            if col_name in df.columns:
                df = df.withColumn(col_name, F.trim(F.col(col_name)))

        # Step 3: Filtering invalid rows and nulls
        logger.info("Step 3: Applying cleaning filters (valid price, valid date, required categoricals)...")
        cleaned_df = df.filter(
            F.col("price").isNotNull() & (F.col("price") > 0) &
            F.col("date").isNotNull() &
            F.col("property_type").isNotNull() & (F.col("property_type") != "") &
            F.col("new_build").isNotNull() & (F.col("new_build") != "") &
            F.col("duration").isNotNull() & (F.col("duration") != "")
        )

        # Fill unknown geographical values
        for col_name in ["town", "district", "county"]:
            if col_name in cleaned_df.columns:
                cleaned_df = cleaned_df.withColumn(
                    col_name,
                    F.when(F.col(col_name).isNull() | (F.col(col_name) == ""), "UNKNOWN").otherwise(F.col(col_name)),
                )

        # Step 4: Remove exact duplicates using primary transaction key
        logger.info("Step 4: Removing exact duplicate records...")
        if "transaction_id" in cleaned_df.columns:
            silver_df = cleaned_df.dropDuplicates(["transaction_id"]).repartition(16)
        else:
            silver_df = cleaned_df.dropDuplicates().repartition(16)

        # Log dataframe info
        log_dataframe_info(silver_df, label="Silver Layer (Cleaned)")

        # Write to Silver Parquet directory
        logger.info(f"Writing Silver Parquet to: {silver_path}")
        silver_path.mkdir(parents=True, exist_ok=True)
        silver_df.write.mode("overwrite").parquet(str(silver_path))

        # Log summary statistics
        final_count = silver_df.count()
        total_dropped = initial_count - final_count
        drop_pct = (total_dropped / initial_count) * 100 if initial_count > 0 else 0
        summary_msg = f"Silver Transformation Summary: Initial={initial_count:,} -> Final={final_count:,} (Dropped={total_dropped:,}, {drop_pct:.2f}%)"
        logger.info(summary_msg)
        transform_log.append(summary_msg)

        elapsed_sec = time.time() - start_time
        transform_log.append(f"Total Processing Time: {elapsed_sec:.2f} seconds")
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(transform_log) + "\n")
        logger.info(f"Transformation log saved to: {log_file_path}")

    except Exception as e:
        logger.error(f"Error during Silver Transformation: {str(e)}", exc_info=True)
        spark.stop()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    run_transformation()
