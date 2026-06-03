"""
02_clustering.py
----------------
Crime clustering script for Perth Crime Hotspot Analysis.

Loads the processed suburb summary CSV and applies K-Means clustering to
group suburbs by their crime profile (type mix and total frequency).
Results are saved to data/processed/ for use by the visualisation script.

Libraries used: Pandas, Scikit-learn, Matplotlib
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SUBURB_SUMMARY_PATH = os.path.join(PROCESSED_DIR, "suburb_crime_summary.csv")
CLUSTERED_PATH = os.path.join(PROCESSED_DIR, "suburb_clusters.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
MAX_K = 8          # maximum number of clusters to evaluate
CHOSEN_K = 4       # final number of clusters (can be overridden by elbow search)


def load_summary(path: str) -> pd.DataFrame:
    """Load the suburb-level crime summary."""
    df = pd.read_csv(path)
    print(f"Loaded suburb summary: {len(df)} suburbs.")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Select and scale numeric crime-type features for clustering.
    Returns the scaled feature matrix and the feature column names.
    """
    exclude = {"suburb", "district", "local_government_area", "latitude", "longitude",
               "total_offences"}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
    print(f"Feature matrix shape: {X_scaled.shape}")
    return X_scaled, feature_cols


def find_optimal_k(X_scaled: pd.DataFrame, max_k: int, figures_dir: str) -> int:
    """
    Evaluate inertia (elbow) and silhouette scores to suggest optimal k.
    Saves an elbow-curve plot and returns the recommended k.
    """
    ks = range(2, max_k + 1)
    inertias, sil_scores = [], []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    # Plot elbow curve
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(ks), inertias, marker="o", color="steelblue")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (Within-cluster SSE)")
    axes[0].set_title("Elbow Curve")
    axes[0].grid(alpha=0.3)

    axes[1].plot(list(ks), sil_scores, marker="s", color="darkorange")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Scores")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    elbow_path = os.path.join(figures_dir, "elbow_curve.png")
    plt.savefig(elbow_path, dpi=150)
    plt.close()
    print(f"Elbow curve saved → {elbow_path}")

    # Pick k with highest silhouette score
    best_k = list(ks)[int(np.argmax(sil_scores))]
    print(f"Best k by silhouette: {best_k}  (score={max(sil_scores):.3f})")
    return best_k


def run_kmeans(X_scaled: pd.DataFrame, k: int) -> np.ndarray:
    """Fit K-Means with the chosen k and return cluster labels."""
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    print(f"K-Means fitted with k={k}. Cluster sizes: {pd.Series(labels).value_counts().to_dict()}")
    return labels


def plot_pca(X_scaled: pd.DataFrame, labels: np.ndarray, k: int, figures_dir: str) -> None:
    """Reduce to 2D with PCA and plot cluster scatter."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = cm.tab10(np.linspace(0, 1, k))
    for cluster_id, color in enumerate(colors):
        mask = labels == cluster_id
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=color, label=f"Cluster {cluster_id}", alpha=0.8, s=80, edgecolors="white")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"K-Means Clusters (k={k}) — PCA projection\n"
                 f"Total variance explained: {explained:.1f}%")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    pca_path = os.path.join(figures_dir, "cluster_pca.png")
    plt.savefig(pca_path, dpi=150)
    plt.close()
    print(f"PCA scatter plot saved → {pca_path}")


def label_clusters(df: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    """
    Assign cluster labels and add human-readable risk tier names based on
    mean total offences per cluster.
    """
    df = df.copy()
    df["cluster"] = labels

    # Build descriptive tier names based on ranked mean total offences
    cluster_means = (
        df.groupby("cluster")["total_offences"].mean()
        .sort_values(ascending=False)
    )
    tiers = ["High Crime", "Moderate-High Crime", "Moderate Crime", "Low Crime"]
    tier_map = {
        cluster_id: tiers[i] if i < len(tiers) else f"Cluster {i}"
        for i, cluster_id in enumerate(cluster_means.index)
    }
    df["risk_tier"] = df["cluster"].map(tier_map)
    print("Risk tier assignment:")
    print(df.groupby(["cluster", "risk_tier"])["total_offences"].mean().to_string())
    return df


def save_results(df_clustered: pd.DataFrame) -> None:
    """Save the clustered suburb data."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_clustered.to_csv(CLUSTERED_PATH, index=False)
    print(f"Clustered data saved → {CLUSTERED_PATH}")


def main() -> None:
    df = load_summary(SUBURB_SUMMARY_PATH)
    X_scaled, feature_cols = prepare_features(df)

    best_k = find_optimal_k(X_scaled, MAX_K, FIGURES_DIR)
    # Allow manual override
    k = CHOSEN_K if CHOSEN_K else best_k

    labels = run_kmeans(X_scaled, k)
    plot_pca(X_scaled, labels, k, FIGURES_DIR)

    df_clustered = label_clusters(df, labels, feature_cols)
    save_results(df_clustered)

    print("\n--- Cluster summary ---")
    print(
        df_clustered.groupby(["cluster", "risk_tier"])["total_offences"]
        .agg(["count", "mean", "min", "max"])
        .rename(columns={"count": "suburbs", "mean": "avg_offences", "min": "min_offences",
                         "max": "max_offences"})
        .to_string()
    )


if __name__ == "__main__":
    main()
