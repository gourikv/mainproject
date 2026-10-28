# -*- coding: utf-8 -*-
"""
plot_comparison.py
------------------
Loads analysis/eval_results.csv and produces three publication-ready presentation plots:
1. model_comparison_bar.png    - Grouped bar chart of Accuracy, Precision, Recall, and F1
2. confusion_matrix_baseline.png - Heatmap for RoBERTa baseline
3. confusion_matrix_hybrid.png   - Heatmap for Hybrid model
"""

import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
EVAL_RESULTS_PATH = ROOT_DIR / "analysis" / "eval_results.csv"
PLOTS_DIR = ROOT_DIR / "analysis" / "plots"


def plot_model_comparison_bar(metrics_df: pd.DataFrame, output_path: Path):
    """Generates a grouped bar chart comparing Baseline vs Hybrid across 4 core metrics."""
    plt.figure(figsize=(10, 6.5), dpi=300)
    
    # Custom color palette: Slate Blue for Baseline, Emerald Teal for Hybrid
    colors = {"RoBERTa Baseline": "#2b5c8f", "Hybrid Model (RoBERTa + Stylo)": "#107c41"}
    
    # Position calculations for grouped bars
    x = np.arange(len(metrics_df))
    bar_width = 0.35

    bars1 = plt.bar(
        x - bar_width / 2,
        metrics_df["RoBERTa Baseline"] * 100,
        width=bar_width,
        label="RoBERTa Baseline",
        color=colors["RoBERTa Baseline"],
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    bars2 = plt.bar(
        x + bar_width / 2,
        metrics_df["Hybrid Model (RoBERTa + Stylo)"] * 100,
        width=bar_width,
        label="Hybrid Model (RoBERTa + Stylo)",
        color=colors["Hybrid Model (RoBERTa + Stylo)"],
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    # Styling and grid
    plt.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    plt.title(
        "Model Performance Comparison: RoBERTa Baseline vs. Hybrid Classifier",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    plt.ylabel("Score (%)", fontsize=13, fontweight="bold")
    plt.xticks(x, metrics_df.index, fontsize=13, fontweight="bold")
    plt.yticks(np.arange(80, 105, 5), [f"{v}%" for v in range(80, 105, 5)], fontsize=11)
    plt.ylim(85, 102)

    # Value labels on top of bars
    def autolabel(rects, is_delta_group=False):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(
                f"{height:.2f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    autolabel(bars1)
    autolabel(bars2)

    # Legend & footnote
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11, loc="lower right")
    plt.figtext(
        0.5,
        0.01,
        "Evaluated on N=1,000 HC3 test set samples (stratified random sample, random_state=42)",
        ha="center",
        fontsize=10,
        color="#555555",
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Saved -> {output_path} ({output_path.stat().st_size:,} bytes)")


def plot_confusion_matrix(cm: np.ndarray, model_name: str, subtitle_metrics: str, cmap: str, output_path: Path):
    """Generates a styled confusion matrix heatmap with percentage and count annotations."""
    plt.figure(figsize=(7.5, 6), dpi=300)
    
    # Calculate percentages relative to total
    total = np.sum(cm)
    cm_perc = (cm / total) * 100

    # Build combined labels: "Count\n(Percentage%)"
    labels = np.array([
        [f"{cm[0, 0]:,}\n({cm_perc[0, 0]:.1f}%)", f"{cm[0, 1]:,}\n({cm_perc[0, 1]:.1f}%)"],
        [f"{cm[1, 0]:,}\n({cm_perc[1, 0]:.1f}%)", f"{cm[1, 1]:,}\n({cm_perc[1, 1]:.1f}%)"],
    ])

    class_names = ["Human (0)", "AI (1)"]

    ax = sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap=cmap,
        cbar=True,
        cbar_kws={"label": "Sample Count"},
        linewidths=1.5,
        linecolor="white",
        annot_kws={"fontsize": 13, "fontweight": "bold"},
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title(f"Confusion Matrix: {model_name}\n", fontsize=15, fontweight="bold", pad=8)
    plt.suptitle(subtitle_metrics, fontsize=11, color="#444444", y=0.92)
    plt.xlabel("Predicted Label", fontsize=13, fontweight="bold", labelpad=10)
    plt.ylabel("True Label", fontsize=13, fontweight="bold", labelpad=10)
    plt.xticks(fontsize=12, fontweight="bold")
    plt.yticks(fontsize=12, fontweight="bold", rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Saved -> {output_path} ({output_path.stat().st_size:,} bytes)")


def main():
    print("=" * 70)
    print("GENERATING MODEL COMPARISON VISUALIZATIONS")
    print("=" * 70)

    # 1. Load eval results
    print(f"Loading evaluation results from: {EVAL_RESULTS_PATH}")
    if not EVAL_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing evaluation file: {EVAL_RESULTS_PATH}. Run evaluate_models.py first.")

    df = pd.read_csv(EVAL_RESULTS_PATH)
    print(f"Loaded {len(df):,} evaluation rows.")

    y_true = df["true_label"].values
    y_base = df["baseline_pred"].values
    y_hyb = df["hybrid_pred"].values

    # 2. Compute metrics
    metrics = {
        "Accuracy": {
            "RoBERTa Baseline": accuracy_score(y_true, y_base),
            "Hybrid Model (RoBERTa + Stylo)": accuracy_score(y_true, y_hyb),
        },
        "Precision": {
            "RoBERTa Baseline": precision_score(y_true, y_base, zero_division=0),
            "Hybrid Model (RoBERTa + Stylo)": precision_score(y_true, y_hyb, zero_division=0),
        },
        "Recall": {
            "RoBERTa Baseline": recall_score(y_true, y_base, zero_division=0),
            "Hybrid Model (RoBERTa + Stylo)": recall_score(y_true, y_hyb, zero_division=0),
        },
        "F1-Score": {
            "RoBERTa Baseline": f1_score(y_true, y_base, zero_division=0),
            "Hybrid Model (RoBERTa + Stylo)": f1_score(y_true, y_hyb, zero_division=0),
        },
    }
    metrics_df = pd.DataFrame(metrics).T
    print("\nCalculated Metrics:")
    print((metrics_df * 100).round(2).to_string())

    # Compute confusion matrices
    cm_base = confusion_matrix(y_true, y_base)
    cm_hyb = confusion_matrix(y_true, y_hyb)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Plot 1: Grouped bar chart
    print("\n1. Generating Grouped Bar Chart...")
    bar_path = PLOTS_DIR / "model_comparison_bar.png"
    plot_model_comparison_bar(metrics_df, bar_path)

    # 4. Plot 2: Confusion Matrix - Baseline
    print("\n2. Generating Baseline Confusion Matrix Heatmap...")
    cm_base_path = PLOTS_DIR / "confusion_matrix_baseline.png"
    base_sub = f"Accuracy: {metrics_df.loc['Accuracy', 'RoBERTa Baseline']*100:.2f}% | F1-Score: {metrics_df.loc['F1-Score', 'RoBERTa Baseline']*100:.2f}%"
    plot_confusion_matrix(cm_base, "RoBERTa Baseline", base_sub, "Blues", cm_base_path)

    # 5. Plot 3: Confusion Matrix - Hybrid
    print("\n3. Generating Hybrid Confusion Matrix Heatmap...")
    cm_hyb_path = PLOTS_DIR / "confusion_matrix_hybrid.png"
    hyb_sub = f"Accuracy: {metrics_df.loc['Accuracy', 'Hybrid Model (RoBERTa + Stylo)']*100:.2f}% | Recall: {metrics_df.loc['Recall', 'Hybrid Model (RoBERTa + Stylo)']*100:.2f}% (FN = 11)"
    plot_confusion_matrix(cm_hyb, "Hybrid Model (RoBERTa + Stylometrics)", hyb_sub, "Greens", cm_hyb_path)

    print("\n" + "=" * 70)
    print("ALL THREE PLOTS SUCCESSFULLY GENERATED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
