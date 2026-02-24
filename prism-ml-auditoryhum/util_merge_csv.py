"""
Merge labels from 2 CSV.
"""

import argparse


def _main(csv1, csv2, csv3):
    """Merge labels from 2 CSV.

    Args:
        csv1: First CSV filepath.
        csv2: Second CSV filepath.
        csv3: output CSV.
    """
    with open(file=csv1, mode="r", encoding="utf-8") as text_file:
        csv1_data = text_file.read()
        csv1_data = csv1_data.split("\n")
    with open(file=csv2, mode="r", encoding="utf-8") as text_file:
        csv2_data = text_file.read()
        csv2_data = csv2_data.split("\n")
        csv2_data = csv2_data + [""] * (len(csv1_data) - len(csv2_data))
    csv3_data = list()
    for index, line in enumerate(csv2_data):
        if line:
            csv3_data.append(line)
        else:
            csv3_data.append(csv1_data[index])
    with open(file=csv3, mode="w", encoding="utf-8") as text_file:
        text_file.write("\n".join(csv3_data))


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="Merge labels from 2 CSV.")
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
        help="output CSV.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
