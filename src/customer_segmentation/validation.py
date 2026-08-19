import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def stability_score(
    X,
    n_clusters,
    n_runs=20,
    sample_fraction=0.8,
    random_state=42,
):
    """Estimate clustering stability with repeated customer subsamples.

    Each run fits K-Means on a random subset and compares assignments on the
    overlap between two runs using Adjusted Rand Index.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    sample_size = max(int(n * sample_fraction), n_clusters * 3)

    assignments = []
    indices = []

    for _ in range(n_runs):
        idx = np.sort(rng.choice(n, size=sample_size, replace=False))
        model = KMeans(
            n_clusters=n_clusters,
            random_state=int(rng.integers(0, 2**31 - 1)),
            n_init=20,
        )
        labels = model.fit_predict(X[idx])
        indices.append(idx)
        assignments.append(labels)

    scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            common, ai, aj = np.intersect1d(
                indices[i], indices[j], return_indices=True
            )
            if len(common) >= n_clusters * 3:
                scores.append(
                    adjusted_rand_score(
                        assignments[i][ai],
                        assignments[j][aj],
                    )
                )

    return {
        "runs": n_runs,
        "sample_fraction": sample_fraction,
        "mean_ari": float(np.mean(scores)) if scores else float("nan"),
        "median_ari": float(np.median(scores)) if scores else float("nan"),
        "min_ari": float(np.min(scores)) if scores else float("nan"),
        "max_ari": float(np.max(scores)) if scores else float("nan"),
        "pairwise_comparisons": len(scores),
    }
