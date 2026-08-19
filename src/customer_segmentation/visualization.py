from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def save_pca_plot(X, labels, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Cluster": labels.astype(str),
    })

    fig, ax = plt.subplots(figsize=(10, 7))
    for cluster, group in plot_df.groupby("Cluster"):
        ax.scatter(group["PC1"], group["PC2"], s=12, alpha=0.55, label=cluster)

    ax.set_title("K-Means Customer Segmentation — PCA Projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Cluster", loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return pca
