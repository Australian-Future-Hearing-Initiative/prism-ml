"""
Cluster data with HDBSCAN.
"""

import argparse
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize


import cluster_kmeans as ck


# HDBSCAN hyperparameters
min_samples = 2
epsilon = 0.0001


def hdbscan_clusters(data, min_cluster_size, min_samples, epsilon):
    """Apply HDBSCAN to data.

    Args:
        data: Input data.
        min_cluster_size: The minimum size of a group to be
        considered a cluster.
        min_samples: Number of samples in a neighborhood for a
        point to be considered a core point.
        epsilon: Hyperparam.

    Return:
        Labels (Noise is labeled -1).
    """
    cluster_instance = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
    )
    labels = cluster_instance.fit_predict(X=data)
    return labels


def _main(cluster_npy, min_cluster_size, label_csv, png_plot):
    """Cluster data with HDBSCAN.

    Args:
        cluster_npy: Path for saved data.
        min_cluster_size: The minimum size of a group to be
        considered a cluster.
        label_csv: Cluster labels saved to CSV.
        png_plot: PNG to save visualisation.
    """
    raw_data = np.load(file=cluster_npy, allow_pickle=True)
    spec_data = raw_data
    # spec_data = normalize(X=spec_data, norm="l2", axis=1)
    labels = hdbscan_clusters(
        data=spec_data,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        epsilon=epsilon,
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
    title = f"t-SNE Visualisation for HDBSCAN "
    title += f"(Found {len(set(labels))} clusters)"
    ck.plot_clusters(
        data=data_reduced,
        labels=labels,
        title=title,
        png_plot=png_plot,
    )
    ck.sil_metrics(data=spec_data, labels=labels)


def _command_line():
    """Process command line arguments."""
    parser = argparse.ArgumentParser(description="Cluster data with HDBSCAN.")
    parser.add_argument(
        "--cluster_npy",
        metavar="S",
        type=str,
        required=True,
        dest="cluster_npy",
        help="Path for saved data.",
    )
    parser.add_argument(
        "--min_cluster_size",
        metavar="N",
        type=int,
        required=True,
        dest="min_cluster_size",
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
    return vars(parser.parse_args())


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
