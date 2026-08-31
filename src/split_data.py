"""
Data Splitting Phase: Train/Test Split on Gold Dataset
Splits gold feature data into 80% train and 20% test partitions with fixed seed (42).
Supports PySpark cluster execution with PyArrow/Pandas engine fallback.
"""

import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import create_spark_session, get_project_root, log_dataframe_info

logger = logging.getLogger("uk-housing-split")


def run_data_split():
    start_time = time.time()
    logger.info("Starting Gold Data Splitting Phase (80/20 Train/Test Split)...")

    root_dir = get_project_root()
    gold_path = root_dir / "data" / "gold" / "gold_features.parquet"
    train_path = root_dir / "data" / "gold" / "train_features.parquet"
    test_path = root_dir / "data" / "gold" / "test_features.parquet"

    if not gold_path.exists():
        logger.error(f"Gold Parquet file not found at: {gold_path}")
        sys.exit(1)

    spark = None
    try:
        spark = create_spark_session(app_name="uk-housing-split")
        logger.info(f"Loading Gold dataset from: {gold_path} with Spark")
        gold_df = spark.read.parquet(str(gold_path))

        train_df, test_df = gold_df.randomSplit([0.8, 0.2], seed=42)
        train_df = train_df.repartition(8)
        test_df = test_df.repartition(4)

        train_count = train_df.count()
        test_count = test_df.count()

        train_df.write.mode("overwrite").parquet(str(train_path))
        test_df.write.mode("overwrite").parquet(str(test_path))

    except Exception as e:
        logger.warning(f"Spark unavailable ({e}). Executing Data Split via Pandas/PyArrow engine...")
        gold_df = pd.read_parquet(gold_path)
        train_df, test_df = train_test_split(gold_df, test_size=0.2, random_state=42)

        train_count = len(train_df)
        test_count = len(test_df)

        if train_path.is_dir():
            shutil.rmtree(train_path)
        if test_path.is_dir():
            shutil.rmtree(test_path)

        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)

    finally:
        if spark:
            spark.stop()

    total_count = train_count + test_count
    logger.info(f"Split Summary (seed=42):")
    logger.info(f"  - Total Rows: {total_count:,}")
    logger.info(f"  - Train Set:  {train_count:,} ({train_count / total_count * 100:.2f}%)")
    logger.info(f"  - Test Set:   {test_count:,} ({test_count / total_count * 100:.2f}%)")

    elapsed_sec = time.time() - start_time
    logger.info(f"Data Splitting completed in {elapsed_sec:.2f} seconds.")


if __name__ == "__main__":
    run_data_split()
