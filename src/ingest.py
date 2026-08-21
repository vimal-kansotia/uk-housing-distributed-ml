"""
Ingestion Phase: Raw CSV -> Bronze Parquet Layer
Loads the UK housing price paid records CSV, repartitions, and stores as raw Bronze Parquet.
"""

import logging
import sys
import time
from pathlib import Path

from pyspark.sql import functions as F

from utils import create_spark_session, get_project_root, log_dataframe_info

logger = logging.getLogger("uk-housing-ingest")


def run_ingestion():
    start_time = time.time()
    logger.info("Starting Bronze Ingestion Phase...")

    root_dir = get_project_root()
    raw_csv_path = root_dir / "data" / "raw" / "price_paid_records.csv"
    bronze_parquet_path = root_dir / "data" / "bronze"

    if not raw_csv_path.exists():
        logger.error(f"Raw CSV file not found at: {raw_csv_path}")
        sys.exit(1)

    spark = create_spark_session(app_name="uk-housing-ingest")

    try:
        logger.info(f"Reading raw CSV from: {raw_csv_path}")
        # Read raw CSV with header inference
        raw_df = (
            spark.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .option("mode", "DROPMALFORMED")
            .load(str(raw_csv_path))
        )

        # Repartition to 8 partitions for distributed processing
        bronze_df = raw_df.repartition(8)

        # Log metadata
        log_dataframe_info(bronze_df, label="Bronze Layer (Raw CSV)")

        logger.info("Displaying 5 sample rows from Bronze dataset:")
        bronze_df.show(5, truncate=False)

        # Write to Bronze Parquet directory
        logger.info(f"Writing Bronze Parquet to: {bronze_parquet_path}")
        bronze_parquet_path.mkdir(parents=True, exist_ok=True)
        bronze_df.write.mode("overwrite").parquet(str(bronze_parquet_path))

        elapsed_sec = time.time() - start_time
        logger.info(f"Bronze Ingestion completed successfully in {elapsed_sec:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error during Bronze Ingestion: {str(e)}", exc_info=True)
        spark.stop()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    run_ingestion()
