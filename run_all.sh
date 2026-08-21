#!/usr/bin/env bash
set -e

echo "======================================================="
echo "   UK HOUSING DISTRIBUTED ML PIPELINE ON SPARK"
echo "======================================================="

echo ""
echo "=== PHASE 1: Data Ingestion (Raw CSV -> Bronze Parquet) ==="
docker exec spark-master python3 /opt/spark/work-dir/src/ingest.py

echo ""
echo "=== PHASE 2: Data Cleaning (Bronze -> Silver Parquet) ==="
docker exec spark-master python3 /opt/spark/work-dir/src/transform.py

echo ""
echo "=== PHASE 3: Feature Engineering & Data Splitting (Silver -> Gold) ==="
docker exec spark-master python3 /opt/spark/work-dir/src/feature_engineering.py
docker exec spark-master python3 /opt/spark/work-dir/src/split_data.py

echo ""
echo "=== PHASE 4: Model Training & Evaluation ==="
echo "--> 4.1 Training Linear Regression..."
docker exec spark-master python3 /opt/spark/work-dir/src/train_linear_regression.py

echo "--> 4.2 Training Random Forest Regressor..."
docker exec spark-master python3 /opt/spark/work-dir/src/train_random_forest.py

echo "--> 4.3 Training Distributed XGBoost..."
docker exec spark-master python3 /opt/spark/work-dir/src/train_xgboost.py

echo "--> 4.4 Training LightGBM Regressor..."
docker exec spark-master python3 /opt/spark/work-dir/src/train_lightgbm.py

echo ""
echo "=== PHASE 5: Aggregating Benchmark Results & Plots ==="
docker exec spark-master python3 /opt/spark/work-dir/src/aggregate_results.py
docker exec spark-master python3 /opt/spark/work-dir/src/plot_benchmark.py

echo ""
echo "======================================================="
echo "       ALL PHASES EXECUTED SUCCESSFULLY! 🎉"
echo "======================================================="

