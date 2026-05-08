import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CLUSTER_FEATURES = ["Age", "Annual_Income", "Spending_Score", "Purchase_Frequency", "Avg_Order_Value"]


def find_optimal_k(df_scaled: pd.DataFrame, k_range: range = range(2, 11)) -> dict:
    features = df_scaled[CLUSTER_FEATURES].values
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(features, labels))
    return {"k_range": list(k_range), "inertias": inertias, "silhouettes": silhouettes}


def run_kmeans(df_scaled: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, KMeans]:
    features = df_scaled[CLUSTER_FEATURES].values
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_scaled = df_scaled.copy()
    df_scaled["Cluster"] = model.fit_predict(features)
    return df_scaled, model


def apply_pca(df_scaled: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    features = df_scaled[CLUSTER_FEATURES].values
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(features)
    df_pca = df_scaled.copy()
    df_pca["PC1"] = components[:, 0]
    df_pca["PC2"] = components[:, 1]
    return df_pca


def get_cluster_summary(df_raw: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    df = df_raw.copy()
    df["Cluster"] = cluster_labels
    summary = df.groupby("Cluster")[CLUSTER_FEATURES].mean().round(2)
    summary["Count"] = df.groupby("Cluster").size()
    return summary
