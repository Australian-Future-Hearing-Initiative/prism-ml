"""
Get Sentence-Transformer embeddings.
"""

import argparse
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def _main(st_model, label_csv, st_npy):
    """Get Sentence-Transformer embeddings.

    Args:
        st_model: Sentence-Transformer model filepath.
        label_csv: Path CSV labels.
        st_npy: Path for Sentence-Transformer embeddings.
    """
    model = SentenceTransformer(st_model)
    model.eval()
    torch.inference_mode(mode=True)
    with open(file=label_csv, mode="r", encoding="utf-8") as label_file:
        content = label_file.read()
        content = content.split("\n")
        content = [x for x in content if x.strip()]
        content = [x.split(",") for x in content]
        content = [x[2].strip() for x in content]
    embeddings = model.encode(sentences=content)
    np.save(file=st_npy, arr=embeddings, allow_pickle=True)


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Get Sentence-Transformer embeddings."
    )
    parser.add_argument(
        "--st_model",
        metavar="S",
        type=str,
        required=True,
        dest="st_model",
        help="Sentence-Transformer model filepath.",
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
        "--st_npy",
        metavar="S",
        type=str,
        required=True,
        dest="st_npy",
        help="Path for Sentence-Transformer embeddings.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
