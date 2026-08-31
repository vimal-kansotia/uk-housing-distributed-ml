"""
Ingestion Phase: Raw CSV -> Bronze Parquet Layer
Loads the UK housing price paid records CSV and stores as raw Bronze Parquet.
Supports PySpark cluster execution with transparent PyArrow/Pandas engine fallback.
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

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

    spark = None
    try:
        spark = create_spark_session(app_name="uk-housing-ingest")
        logger.info(f"Reading raw CSV with Spark from: {raw_csv_path}")
        raw_df = (
            spark.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .option("mode", "DROPMALFORMED")
            .load(str(raw_csv_path))
        )
        bronze_df = raw_df.repartition(8)
        log_dataframe_info(bronze_df, label="Bronze Layer (Raw CSV)")
        bronze_parquet_path.mkdir(parents=True, exist_ok=True)
        bronze_df.write.mode("overwrite").parquet(str(bronze_parquet_path))

    except Exception as e:
        logger.warning(f"Spark unavailable ({e}). Executing via high-performance PyArrow engine...")
        bronze_parquet_path.mkdir(parents=True, exist_ok=True)
        pdf = pd.read_csv(raw_csv_path)
        logger.info(f"Loaded {len(pdf):,} records from raw CSV.")
        pdf.to_parquet(bronze_parquet_path / "part-00000.parquet", index=False)
        logger.info(f"Wrote Bronze Parquet to {bronze_parquet_path}")

    finally:
        if spark:
            spark.stop()

    elapsed_sec = time.time() - start_time
    logger.info(f"Bronze Ingestion completed successfully in {elapsed_sec:.2f} seconds.")


if __name__ == "__main__":
    run_ingestion()
