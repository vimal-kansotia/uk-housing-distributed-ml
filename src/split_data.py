"""
Data Splitting Phase: Train/Test Split on Gold Dataset
Splits gold feature data into 80% train and 20% test partitions with fixed seed (42).
"""

import logging
import sys
import time
from pathlib import Path

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

    spark = create_spark_session(app_name="uk-housing-split")

    try:
        logger.info(f"Loading Gold dataset from: {gold_path}")
        gold_df = spark.read.parquet(str(gold_path))

        # Split 80% train, 20% test with seed=42
        train_df, test_df = gold_df.randomSplit([0.8, 0.2], seed=42)

        # Repartition for balanced distributed processing across cluster workers
        train_df = train_df.repartition(8)
        test_df = test_df.repartition(4)

        train_count = train_df.count()
        test_count = test_df.count()
        total_count = train_count + test_count

        logger.info(f"Split Summary (seed=42):")
        logger.info(f"  - Total Rows: {total_count:,}")
        logger.info(f"  - Train Set:  {train_count:,} ({train_count / total_count * 100:.2f}%)")
        logger.info(f"  - Test Set:   {test_count:,} ({test_count / total_count * 100:.2f}%)")

        logger.info(f"Writing Train Parquet to: {train_path}")
        train_df.write.mode("overwrite").parquet(str(train_path))

        logger.info(f"Writing Test Parquet to: {test_path}")
        test_df.write.mode("overwrite").parquet(str(test_path))

        elapsed_sec = time.time() - start_time
        logger.info(f"Data Splitting completed in {elapsed_sec:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error during Data Splitting: {str(e)}", exc_info=True)
        spark.stop()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    run_data_split()
