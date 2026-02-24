"""
Cluster data with Spectral.
"""

import argparse
import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import normalize


import cluster_kmeans as ck


# Spectral Clustering hyperparameters
affinity = "nearest_neighbors"
n_neighbors = 30
n_components = 2


def spectral_embedding(data, n_components, affinity, n_neighbors, seed):
    """Create spectral embeddings.

    Args:
        data: Input data.
        affinity: Type of distance calculation.
        n_neighbors: Neighbours when doing Eigen decomposition.
        n_components: Reduce the affinity matrix to
        this number of dimensions.
        seed: Random seed.

    Returns:
        Spectral embeddings.
    """
    embedder = SpectralEmbedding(
        n_components=n_components,
        affinity=affinity,
        n_neighbors=n_neighbors,
        random_state=seed,
    )
    spectral_embedding_data = embedder.fit_transform(X=data)
    return spectral_embedding_data


def _main(cluster_npy, n_clusters, label_csv, png_plot):
    """Cluster data with Spectral.

    Args:
        cluster_npy: Path for saved MFCC.
        n_clusters: Number of clusters.
        label_csv: Cluster labels saved to CSV.
        png_plot: PNG to save visualisation.
    """
    raw_data = np.load(file=cluster_npy, allow_pickle=True)
    spec_data = spectral_embedding(
        data=raw_data,
        n_components=n_components,
        affinity=affinity,
        n_neighbors=n_neighbors,
        seed=ck.seed,
    )
    # spec_data = normalize(X=spec_data, norm="l2", axis=1)
    labels = ck.kmeans_clusters(
        data=spec_data,
        n_clusters=n_clusters,
        init=ck.kmeans_init,
        max_iter=ck.kmeans_max_iter,
        seed=ck.seed,
    )
    np.savetxt(fname=label_csv, X=labels, delimiter=",", fmt="%d")
    data_reduced = ck.tsne_reduce(
        data=raw_data,
        n_components=ck.tsne_n_components,
        perplexity=ck.perplexity,
        early_exaggeration=ck.early_exaggeration,
        learning_rate=ck.learning_rate,
        max_iter=ck.tsne_max_iter,
        n_iter_without_progress=ck.n_iter_without_progress,
        metric=ck.metric,
        init=ck.tsne_init,
        random_state=ck.seed,
    )
    ck.plot_clusters(
        data=data_reduced,
        labels=labels,
        title=f"t-SNE Visualisation Spectral Clustering k={n_clusters}",
        png_plot=png_plot,
    )
    ck.sil_metrics(data=spec_data, labels=labels)


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="Cluster data with Spectral.")
    parser.add_argument(
        "--cluster_npy",
        metavar="S",
        type=str,
        required=True,
        dest="cluster_npy",
        help="Path for saved data.",
    )
    parser.add_argument(
        "--n_clusters",
        metavar="N",
        type=int,
        required=True,
        dest="n_clusters",
        help="Number of clusters.",
    )
    parser.add_argument(
        "--label_csv",
        metavar="S",
        type=str,
        required=True,
        dest="label_csv",
        help="Cluster labels saved to CSV.",
    )
    parser.add_argument(
        "--png_plot",
        metavar="S",
        type=str,
        required=True,
        dest="png_plot",
        help="PNG to save visualisation.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
