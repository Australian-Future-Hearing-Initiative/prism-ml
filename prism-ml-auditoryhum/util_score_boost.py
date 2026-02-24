"""
Check CSV scores.
"""

import argparse
import numpy as np


def _main(csv1, csv2, csv3):
    """Check CSV scores. Check the score
    difference for specified positions
    in two different CSVs.

    Args:
        csv1: First CSV filepath.
        csv2: Second CSV filepath.
        csv3: CSV filepath representing the rows
        which need to be checked.
    """
    with open(file=csv1, mode="r", encoding="utf-8") as text_file:
        csv1_data = text_file.read()
        csv1_data = csv1_data.split("\n")
    with open(file=csv2, mode="r", encoding="utf-8") as text_file:
        csv2_data = text_file.read()
        csv2_data = csv2_data.split("\n")
    with open(file=csv3, mode="r", encoding="utf-8") as text_file:
        csv3_data = text_file.read()
        csv3_data = csv3_data.split("\n")
    csv1_scores = list()
    csv2_scores = list()
    for index, line in enumerate(csv3_data):
        if line:
            csv1_scores.append(float(csv1_data[index].split(",")[3]))
            csv2_scores.append(float(csv2_data[index].split(",")[3]))
    print("Difference scores MLLM vs MLLM + human")
    print(np.mean(a=csv1_scores))
    print(np.mean(a=csv2_scores))


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="Check CSV scores.")
    parser.add_argument(
        "--csv1",
        metavar="S",
        type=str,
        required=True,
        dest="csv1",
        help="First CSV filepath.",
    )
    parser.add_argument(
        "--csv2",
        metavar="S",
        type=str,
        required=True,
        dest="csv2",
        help="Second CSV filepath.",
    )
    parser.add_argument(
        "--csv3",
        metavar="S",
        type=str,
        required=True,
        dest="csv3",
        help="Check positions.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
