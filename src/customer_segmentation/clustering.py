import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def safe_silhouette(X, labels, sample_size=None, random_state=42):
    """Safely calculates the silhouette score, ignoring noise (-1) and single clusters."""
    labels = np.asarray(labels)
    unique = set(labels)
    if len(unique) < 2:
        return float("nan")

    # Filter out noise points (e.g., DBSCAN noise label -1)
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return float("nan")

    return float(
        silhouette_score(
            X[mask],
            labels[mask],
            sample_size=sample_size,
            random_state=random_state,
        )
    )


def evaluate_kmeans(X, k_values, random_state=42, sample_size=None):
    """Evaluates K-Means clustering across a range of k values."""
    rows = []
    for k in k_values:
        model = fit_kmeans(X, n_clusters=k, random_state=random_state)
        labels = model.labels_

        score = safe_silhouette(
            X, labels, sample_size=sample_size, random_state=random_state
        )

        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": score,
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans(X, n_clusters, random_state=42):
    """Fits a K-Means model on the input data."""
    return KMeans(
        n_clusters=int(n_clusters),
        random_state=random_state,
        n_init=20,
    ).fit(X)


def fit_gmm(X, n_components, random_state=42):
    """Fits a Gaussian Mixture Model on the input data."""
    return GaussianMixture(
        n_components=int(n_components),
        covariance_type="full",
        random_state=random_state,
        n_init=5,
    ).fit(X)


def fit_dbscan(X, eps=0.5, min_samples=10):
    """Fits a DBSCAN model on the input data."""
    return DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit(X)