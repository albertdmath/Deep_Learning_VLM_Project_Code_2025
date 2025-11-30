import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
ANALYSIS_CSV = Path("attention_metrics.csv")
OUTPUT_FOLDER = Path("visualizations")
OUTPUT_FOLDER.mkdir(exist_ok=True)

def plot_layer_trends(df):
    """
    Generates a dual-axis line chart showing mean IoU and Entropy per layer.
    Also includes standard deviation shading.
    """
    # Group by layer and calculate mean and std
    layer_stats = df.groupby("layer").agg({
        "iou": ["mean", "std"],
        "entropy": ["mean", "std"]
    })
    
    layers = layer_stats.index
    iou_mean = layer_stats["iou"]["mean"]
    iou_std = layer_stats["iou"]["std"]
    ent_mean = layer_stats["entropy"]["mean"]
    ent_std = layer_stats["entropy"]["std"]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Plot IoU on Left Y-Axis ---
    color = 'tab:blue'
    ax1.set_xlabel('Layer Index', fontsize=12)
    ax1.set_ylabel('Average IoU (Higher is Better)', color=color, fontsize=12)
    ax1.plot(layers, iou_mean, color=color, marker='o', label='Mean IoU')
    ax1.fill_between(layers, iou_mean - iou_std, iou_mean + iou_std, color=color, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # --- Plot Entropy on Right Y-Axis ---
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    ax2.set_ylabel('Average Entropy (Lower is Sharper)', color=color, fontsize=12)
    ax2.plot(layers, ent_mean, color=color, marker='s', linestyle='--', label='Mean Entropy')
    ax2.fill_between(layers, ent_mean - ent_std, ent_mean + ent_std, color=color, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and Layout
    plt.title('Layer-wise Attention Evolution: IoU vs. Entropy', fontsize=14)
    fig.tight_layout()
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    output_path = OUTPUT_FOLDER / "layer_iou_entropy_chart.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved trend plot to {output_path}")

def plot_iou_boxplots(df):
    """
    Generates box plots for IoU distribution per layer to visualize variance.
    """
    plt.figure(figsize=(14, 7))
    sns.boxplot(x="layer", y="iou", data=df, palette="viridis")
    
    plt.title("Distribution of IoU Scores per Layer", fontsize=14)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("IoU Score", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    output_path = OUTPUT_FOLDER / "layer_iou_boxplots.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved box plot to {output_path}")

def main():
    if not ANALYSIS_CSV.exists():
        print(f"Error: {ANALYSIS_CSV} not found. Run attention_analysis.py first.")
        return

    print("Loading analysis data...")
    df = pd.read_csv(ANALYSIS_CSV)
    
    if df.empty:
        print("CSV is empty.")
        return

    print(f"Loaded {len(df)} records. Generating plots...")
    
    # 1. Line Chart (Trends)
    plot_layer_trends(df)
    
    # 2. Box Plots (Variance)
    plot_iou_boxplots(df)
    
    print("\nStatistics visualization complete!")

if __name__ == "__main__":
    main()