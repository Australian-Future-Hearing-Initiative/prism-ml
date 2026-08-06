"""
Calculate Class Vector.
"""

import argparse
import glob
import librosa
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ClapModel, ClapProcessor


import composite_label_stats as cls
import features_clap_text as fct
import features_clap as fc

# CLAP hyperparameters
# device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_map = "auto"


def _main(filedir, clap_model, scores_csv, cvector_npy):
    """Calculate Class Vector.

    Args:
        filedir: Regex file path of audio. E.g. "*.wav".
        clap_model: CLAP model filepath.
        scores_csv: CSV file path containing top scoring labels.
        cvector_npy: Path for saving class vector.
    """
    filelist = glob.glob(pathname=filedir)
    filelist.sort()
    processor = ClapProcessor.from_pretrained(
        pretrained_model_name_or_path=clap_model
    )
    model = ClapModel.from_pretrained(
        pretrained_model_name_or_path=clap_model,
        device_map=device_map,
    )
    model.eval()
    torch.inference_mode(mode=True)
    # Get label embeddings
    label_list = cls.get_top_labels(top_label_scores_csv=scores_csv)
    unique_labels = list(set(label_list))
    unique_labels.sort()
    label_embeddings_list = fct.text_label_embeddings(
        model=model,
        processor=processor,
        text_label=unique_labels,
    )
    # Get audio embeddings
    audio_embeddings_list = fc.get_audio_embeddings(
        filelist=filelist, model=model, processor=processor
    )
    # Compute the cosine scores
    text_embeddings = np.concatenate(label_embeddings_list, axis=0)
    audio_embeddings = np.concatenate(audio_embeddings_list, axis=0)
    # The cosine scores are used a a class vector
    cosine_scores = cosine_similarity(X=audio_embeddings, Y=text_embeddings)
    np.save(file=cvector_npy, arr=cosine_scores, allow_pickle=True)
    #print(unique_labels)


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate Class Vector CLAP embeddings."
    )
    parser.add_argument(
        "--filedir",
        metavar="S",
        type=str,
        required=True,
        dest="filedir",
        help='Regex file path of audio. E.g. "*.wav".',
    )
    parser.add_argument(
        "--clap_model",
        metavar="S",
        type=str,
        required=True,
        dest="clap_model",
        help="CLAP model filepath.",
    )
    parser.add_argument(
        "--scores_csv",
        metavar="S",
        type=str,
        required=True,
        dest="scores_csv",
        help="CSV file path containing top scoring labels.",
    )
    parser.add_argument(
        "--cvector_npy",
        metavar="S",
        type=str,
        required=True,
        dest="cvector_npy",
        help="Path for saving class vector.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
