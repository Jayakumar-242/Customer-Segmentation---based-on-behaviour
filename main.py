import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import preprocess
from src.clustering import find_optimal_k, run_kmeans, apply_pca, get_cluster_summary
from src.visualization import plot_elbow, plot_cluster_scatter, plot_cluster_profiles, plot_distribution
from src.utils import label_segments, save_segmented_output, print_summary

N_CLUSTERS = 4


def main():
    print("Loading and preprocessing data...")
    df_raw, df_scaled, scaler = preprocess("data/customers.csv")

    print("Finding optimal clusters...")
    elbow_data = find_optimal_k(df_scaled)
    plot_elbow(elbow_data)

    print(f"Running KMeans with k={N_CLUSTERS}...")
    df_clustered, model = run_kmeans(df_scaled, n_clusters=N_CLUSTERS)

    df_raw["Cluster"] = df_clustered["Cluster"].values
    df_pca = apply_pca(df_clustered)
    df_pca["Cluster"] = df_clustered["Cluster"].values

    summary = get_cluster_summary(df_raw, df_raw["Cluster"].values)
    print_summary(summary)

    print("Generating visualizations...")
    plot_cluster_scatter(df_pca)
    plot_cluster_profiles(summary)
    plot_distribution(df_raw)

    df_labeled = label_segments(df_raw)
    save_segmented_output(df_labeled)
    print("Pipeline complete. Charts saved to outputs/charts/")


if __name__ == "__main__":
    main()
