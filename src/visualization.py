import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

# ── Professional palette ──────────────────────────────────────────────────────
PALETTE = ["#2E4057", "#048A81", "#54C6EB", "#EF946C", "#C4A35A", "#8338EC", "#FF006E"]
FONT = "DejaVu Sans"
BG = "#F7F9FC"
GRID = "#E0E6EF"

mpl.rcParams.update({
    "font.family": FONT,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

os.makedirs("outputs/charts", exist_ok=True)


def plot_elbow(elbow_data: dict, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Optimal Number of Clusters", fontsize=16, fontweight="bold", color="#2E4057")

    axes[0].plot(elbow_data["k_range"], elbow_data["inertias"], marker="o", color=PALETTE[0], linewidth=2.5)
    axes[0].set_title("Elbow Method — Inertia")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(elbow_data["k_range"], elbow_data["silhouettes"], marker="s", color=PALETTE[1], linewidth=2.5)
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    if save:
        plt.savefig("outputs/charts/elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_cluster_scatter(df_pca: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, cluster in enumerate(sorted(df_pca["Cluster"].unique())):
        mask = df_pca["Cluster"] == cluster
        ax.scatter(df_pca.loc[mask, "PC1"], df_pca.loc[mask, "PC2"],
                   label=f"Segment {cluster}", color=PALETTE[i % len(PALETTE)],
                   alpha=0.75, s=55, edgecolors="white", linewidths=0.4)
    ax.set_title("Customer Segments — PCA Projection", fontsize=15, fontweight="bold", color="#2E4057")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Segment", framealpha=0.9)
    plt.tight_layout()
    if save:
        plt.savefig("outputs/charts/cluster_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_cluster_profiles(summary: pd.DataFrame, save: bool = True):
    features = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency", "Avg_Order_Value"]
    fig, axes = plt.subplots(1, len(features), figsize=(18, 5))
    fig.suptitle("Cluster Feature Profiles", fontsize=15, fontweight="bold", color="#2E4057")
    for ax, feat in zip(axes, features):
        bars = ax.bar(summary.index.astype(str), summary[feat],
                      color=[PALETTE[i % len(PALETTE)] for i in range(len(summary))],
                      edgecolor="white", linewidth=0.8)
        ax.set_title(feat.replace("_", " "))
        ax.set_xlabel("Segment")
    plt.tight_layout()
    if save:
        plt.savefig("outputs/charts/cluster_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_distribution(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Feature Distributions by Segment", fontsize=15, fontweight="bold", color="#2E4057")
    features = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency", "Avg_Order_Value"]
    axes_flat = axes.flatten()
    for i, feat in enumerate(features):
        for j, cluster in enumerate(sorted(df["Cluster"].unique())):
            axes_flat[i].hist(df.loc[df["Cluster"] == cluster, feat], bins=20,
                              alpha=0.65, color=PALETTE[j % len(PALETTE)], label=f"Seg {cluster}")
        axes_flat[i].set_title(feat.replace("_", " "))
        axes_flat[i].legend(fontsize=7)
    axes_flat[-1].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("outputs/charts/distributions.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_interactive_3d(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter_3d(
        df, x="Annual_Income", y="Spending_Score", z="Age",
        color=df["Cluster"].astype(str),
        color_discrete_sequence=PALETTE,
        title="3D Customer Segmentation",
        labels={"color": "Segment"},
        opacity=0.8,
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        font_family=FONT,
        title_font_size=18,
        title_font_color="#2E4057",
        legend_title_text="Segment"
    )
    return fig
