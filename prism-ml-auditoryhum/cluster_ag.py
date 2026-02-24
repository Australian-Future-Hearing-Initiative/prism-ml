"""
Cluster data with Agglomerative Clustering.
"""

import argparse
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


import cluster_kmeans as ck


# Agglomerative Clustering hyperparameters
metric = "euclidean"
linkage = "ward"


def ag_clusters(data, n_clusters, metric, linkage):
    """Apply Agglomerative Clustering to data.

    Args:
        data: Input data.
        n_clusters: Number of clusters.
        metric: Distance metric.
        linkage: AG hyperparam.

    Return:
        Labels.
    """
    cluster_instance = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        linkage=linkage,
    )
    labels = cluster_instance.fit_predict(X=data)
    return labels


def _main(cluster_npy, n_clusters, label_csv, png_plot):
    """Cluster data with Agglomerative Clustering.

    Args:
        cluster_npy: Path for saved data.
        n_clusters: Number of clusters.
        label_csv: Cluster labels saved to CSV.
        png_plot: PNG to save visualisation.
    """
    raw_data = np.load(file=cluster_npy, allow_pickle=True)
    spec_data = raw_data
    # spec_data = normalize(X=spec_data, norm="l2", axis=1)
    labels = ag_clusters(
        data=spec_data,
        n_clusters=n_clusters,
        metric=metric,
        linkage=linkage,
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
    title = f"t-SNE Visualisation for Agglomerative Clustering "
    title += f"k={n_clusters}"
    ck.plot_clusters(
        data=data_reduced,
        labels=labels,
        title=title,
        png_plot=png_plot,
    )
    ck.sil_metrics(data=spec_data, labels=labels)


def _command_line():
    """Process command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cluster data with Agglomerative Clustering."
    )
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
    return vars(parser.parse_args())


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
