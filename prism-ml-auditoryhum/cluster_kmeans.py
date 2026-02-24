"""
Cluster data with KMEANS.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


# Shared parameters
seed = 100
# KMeans hyperparameters
kmeans_init = "k-means++"
kmeans_max_iter = 300
# t-SNE hyperparameters
tsne_n_components = 2
perplexity = 35.0
early_exaggeration = 12.0
learning_rate = "auto"
tsne_max_iter = 500
n_iter_without_progress = 100
metric = "euclidean"
tsne_init = "random"


def kmeans_clusters(data, n_clusters, init, max_iter, seed):
    """Apply K-Means to data.

    Args:
        data: Input data.
        n_clusters: Number of clusters.
        init: K-Means init.
        max_iter: K-Means max iterations.
        seed: K-Means random seed.

    Return:
        Labels.
    """
    cluster_instance = KMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        random_state=seed,
        verbose=0,
    )
    cluster_instance.fit(X=data)
    labels = cluster_instance.labels_
    return labels


def tsne_reduce(
    data,
    n_components,
    perplexity,
    early_exaggeration,
    learning_rate,
    max_iter,
    n_iter_without_progress,
    metric,
    init,
    random_state,
):
    """Reduce data dimensions using t-SNE.

    Args:
        data: Input data.
        n_components: Reduce to this many dimensions.
        perplexity: Calculate new dim values based on this
        many neighbours.
        early_exaggeration: Cluster tightness.
        learning_rate: Learning rate.
        max_iter: Max iterations.
        n_iter_without_progress: Termination condition.
        metric: Distance metric.
        init: Type of initialisation.
        random_state: Random seed.

    Return:
        Data after dimension reduction.
    """
    tsne_instance = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        max_iter=max_iter,
        n_iter_without_progress=n_iter_without_progress,
        metric=metric,
        init=init,
        random_state=seed,
    )
    data_reduce = tsne_instance.fit_transform(X=data)
    return data_reduce


def plot_clusters(data, labels, title, png_plot):
    """Plot clusters.

    Args:
        data: Plot data.
        labels: Labels.
        title: Title of the plot.
        png_plot: Save scatter plot PNG image to this file.
    """
    plt.figure(figsize=(12, 12))
    plt.scatter(x=data[:, 0], y=data[:, 1], c=labels)
    plt.title(label=title, fontsize=35, fontweight="bold", wrap=True)
    plt.xlabel(xlabel="X-axis", fontsize=35, fontweight="bold")
    plt.ylabel(ylabel="Y-axis", fontsize=35, fontweight="bold")
    plt.tick_params(axis="both", labelsize=25)
    plt.savefig(fname=png_plot, format="png")


def sil_metrics(data, labels):
    """Show Silhouette Score metrics.

    Args:
        data: Input data.
        labels: Input labels.
    """
    sil_score = silhouette_score(X=data, labels=labels)
    print("Silhouette Score", sil_score)


def _main(cluster_npy, n_clusters, label_csv, png_plot):
    """Cluster data with KMEANS.

    Args:
        cluster_npy: Path for saved data.
        n_clusters: Number of clusters.
        label_csv: Cluster labels saved to CSV.
        png_plot: PNG to save visualisation.
    """
    raw_data = np.load(file=cluster_npy, allow_pickle=True)
    spec_data = raw_data
    # spec_data = normalize(X=spec_data, norm="l2", axis=1)
    labels = kmeans_clusters(
        data=spec_data,
        n_clusters=n_clusters,
        init=kmeans_init,
        max_iter=kmeans_max_iter,
        seed=seed,
    )
    np.savetxt(fname=label_csv, X=labels, delimiter=",", fmt="%d")
    data_reduced = tsne_reduce(
        data=raw_data,
        n_components=tsne_n_components,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        max_iter=tsne_max_iter,
        n_iter_without_progress=n_iter_without_progress,
        metric=metric,
        init=tsne_init,
        random_state=seed,
    )
    plot_clusters(
        data=data_reduced,
        labels=labels,
        title=f"t-SNE Visualisation K-Means k={n_clusters}",
        png_plot=png_plot,
    )
    sil_metrics(data=spec_data, labels=labels)


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="Cluster data with KMEANS.")
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
