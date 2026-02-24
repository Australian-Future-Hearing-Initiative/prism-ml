"""
Get stats for composite labels.
"""

import argparse
import numpy as np


def _label_distro_vector(top_label, cluster_indices):
    """Print the label distribution vector for every cluster.

    Args:
        top_label: Array of labels.
        cluster_indices: Array of cluster indices.
    """
    prompt = "Provide a short sentence to describe this set "
    prompt += "of audio samples. The frequency distribution "
    prompt += "of individual labels for this set of audio "
    prompt += "samples is provided:"
    for single_index in set(cluster_indices):
        sub_label = top_label[cluster_indices == single_index]
        print(prompt)
        for single_label in set(sub_label):
            print(f"{single_label}, {sum(single_label == sub_label)}")
        print(f"Total samples: {len(sub_label)}")
        print("")


def _cluster_distro_vector(top_label, cluster_indices):
    """Print the cluster distribution vector for every label.

    Args:
        top_label: Array of labels.
        cluster_indices: Array of cluster indices.
    """
    cluster_counter = 0
    for single_label in set(top_label):
        sub_cluster = cluster_indices[top_label == single_label]
        line = f"{single_label}, belongs to this many clusters: "
        line += f"{len(set(sub_cluster))}"
        print(line)
        if len(set(sub_cluster)) > 1:
            cluster_counter += 1
    print(f"Total labels: {len(set(top_label))}")
    print(f"Total labels belonging to more than 1 cluster: {cluster_counter}")
    print("")


def _main(label_csv, top_label_scores_csv):
    """Get stats for composite labels.

    Args:
        label_csv: Path CSV labels.
        top_label_scores_csv: Path for saving scores.
    """
    with open(
        file=top_label_scores_csv, mode="r", encoding="utf-8"
    ) as label_file:
        top_label = label_file.read()
        top_label = top_label.split("\n")
        top_label = [x for x in top_label if x.strip()]
        top_label = [x.split(",") for x in top_label]
        top_label = [x[2].strip() for x in top_label]
    with open(file=label_csv, mode="r", encoding="utf-8") as label_file:
        cluster_indices = label_file.read()
        cluster_indices = cluster_indices.split("\n")
        cluster_indices = [x for x in cluster_indices if x.strip()]
        cluster_indices = [int(x) for x in cluster_indices]
    top_label = np.array(object=top_label)
    cluster_indices = np.array(object=cluster_indices)
    _label_distro_vector(
        top_label=top_label, cluster_indices=cluster_indices
    )
    _cluster_distro_vector(
        top_label=top_label, cluster_indices=cluster_indices
    )


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Get stats for composite labels."
    )
    parser.add_argument(
        "--label_csv",
        metavar="S",
        type=str,
        required=True,
        dest="label_csv",
        help="Path CSV labels.",
    )
    parser.add_argument(
        "--top_label_scores_csv",
        metavar="S",
        type=str,
        required=True,
        dest="top_label_scores_csv",
        help="Path for saving chosen labels and scores.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
