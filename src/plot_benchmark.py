"""
Generate Benchmark Plots for UK Housing Price Prediction.
Saves high-resolution plots to results/plots/.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots():
    root_dir = Path(__file__).resolve().parent.parent if not Path("/opt/spark/work-dir").exists() else Path("/opt/spark/work-dir")
    csv_path = root_dir / "results" / "model_comparison.csv"
    plots_dir = root_dir / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print("Generating benchmark plots for:")
    print(df)

    # Set aesthetic style
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    # 1. MAE Comparison
    plt.figure(figsize=(9, 5))
    bars = sns.barplot(x="model", y="mae", data=df, palette=palette)
    plt.title("Model Comparison: Mean Absolute Error (MAE in £)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Model", fontweight="bold")
    plt.ylabel("MAE (£)", fontweight="bold")
    plt.ylim(0, df["mae"].max() * 1.15)
    for bar in bars.patches:
        val = bar.get_height()
        bars.annotate(f"£{val:,.2f}",
                      (bar.get_x() + bar.get_width() / 2, val),
                      ha="center", va="bottom", fontsize=11, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(plots_dir / "mae_comparison.png", dpi=300)
    plt.close()

    # 2. RMSE Comparison
    plt.figure(figsize=(9, 5))
    bars = sns.barplot(x="model", y="rmse", data=df, palette=palette)
    plt.title("Model Comparison: Root Mean Squared Error (RMSE in £)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Model", fontweight="bold")
    plt.ylabel("RMSE (£)", fontweight="bold")
    plt.ylim(0, df["rmse"].max() * 1.15)
    for bar in bars.patches:
        val = bar.get_height()
        bars.annotate(f"£{val:,.2f}",
                      (bar.get_x() + bar.get_width() / 2, val),
                      ha="center", va="bottom", fontsize=11, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(plots_dir / "rmse_comparison.png", dpi=300)
    plt.close()

    # 3. R² Score Comparison
    plt.figure(figsize=(9, 5))
    bars = sns.barplot(x="model", y="r2", data=df, palette=palette)
    plt.title("Model Comparison: Coefficient of Determination (R² Score)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Model", fontweight="bold")
    plt.ylabel("R² Score", fontweight="bold")
    plt.ylim(0, max(0.3, df["r2"].max() * 1.25))
    for bar in bars.patches:
        val = bar.get_height()
        bars.annotate(f"{val:.4f}",
                      (bar.get_x() + bar.get_width() / 2, val),
                      ha="center", va="bottom", fontsize=11, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(plots_dir / "r2_comparison.png", dpi=300)
    plt.close()

    # 4. Training Time Comparison
    plt.figure(figsize=(9, 5))
    bars = sns.barplot(x="model", y="training_time_sec", data=df, palette=palette)
    plt.title("Model Training Time (seconds)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Model", fontweight="bold")
    plt.ylabel("Training Time (s)", fontweight="bold")
    plt.ylim(0, df["training_time_sec"].max() * 1.15)
    for bar in bars.patches:
        val = bar.get_height()
        ann_text = f"{val:.3f}s" if val < 0.05 else f"{val:,.2f}s"
        bars.annotate(ann_text,
                      (bar.get_x() + bar.get_width() / 2, val),
                      ha="center", va="bottom", fontsize=11, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(plots_dir / "training_time_comparison.png", dpi=300)
    plt.close()

    # 5. Prediction Latency Comparison
    plt.figure(figsize=(9, 5))
    bars = sns.barplot(x="model", y="prediction_time_sec", data=df, palette=palette)
    plt.title("Prediction Latency on Test Set (seconds)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Model", fontweight="bold")
    plt.ylabel("Prediction Time (s)", fontweight="bold")
    plt.ylim(0, df["prediction_time_sec"].max() * 1.25)
    for bar in bars.patches:
        val = bar.get_height()
        ann_text = f"{val:.3f}s" if val < 0.05 else f"{val:.2f}s"
        bars.annotate(ann_text,
                      (bar.get_x() + bar.get_width() / 2, val),
                      ha="center", va="bottom", fontsize=11, fontweight="bold", xytext=(0, 4), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(plots_dir / "prediction_latency_comparison.png", dpi=300)
    plt.close()

    # 6. Combined 2x2 Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # Subplot 1: MAE
    sns.barplot(ax=axes[0, 0], x="model", y="mae", data=df, palette=palette)
    axes[0, 0].set_title("Mean Absolute Error (MAE in £)", fontweight="bold")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("MAE (£)")

    # Subplot 2: RMSE
    sns.barplot(ax=axes[0, 1], x="model", y="rmse", data=df, palette=palette)
    axes[0, 1].set_title("Root Mean Squared Error (RMSE in £)", fontweight="bold")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("RMSE (£)")

    # Subplot 3: R²
    sns.barplot(ax=axes[1, 0], x="model", y="r2", data=df, palette=palette)
    axes[1, 0].set_title("R² Score (Higher is Better)", fontweight="bold")
    axes[1, 0].set_xlabel("Model", fontweight="bold")
    axes[1, 0].set_ylabel("R²")

    # Subplot 4: Training Time
    sns.barplot(ax=axes[1, 1], x="model", y="training_time_sec", data=df, palette=palette)
    axes[1, 1].set_title("Training Time (seconds)", fontweight="bold")
    axes[1, 1].set_xlabel("Model", fontweight="bold")
    axes[1, 1].set_ylabel("Seconds")

    plt.suptitle("UK Housing Price Prediction — Distributed ML Benchmark Summary", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison_dashboard.png", dpi=300)
    plt.close()

    print(f"All plots successfully saved to {plots_dir}")

if __name__ == "__main__":
    generate_plots()
