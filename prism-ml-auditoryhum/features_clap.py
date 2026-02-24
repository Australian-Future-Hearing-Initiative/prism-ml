"""
Calculate CLAP embeddings.
"""

import argparse
import glob
import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor


# CLAP hyperparameters
device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _main(filedir, clap_model, clap_npy):
    """Calculate CLAP embeddings.

    Args:
        filedir: Regex file path of audio. E.g. "*.wav".
        clap_model: CLAP model filepath.
        clap_npy: Path for saving CLAP embeddings.
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
    total = len(filelist)
    list_of_embeddings = list()
    for index, filename in enumerate(filelist):
        y, _ = librosa.load(
            path=filename,
            sr=processor.feature_extractor.sampling_rate,
            mono=True,
        )
        inputs = processor(
            audio=y,
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
        ).to(device=model.device)
        audio_embeddings = model.get_audio_features(**inputs)
        audio_embeddings = audio_embeddings.detach().cpu().float().numpy()
        list_of_embeddings.append(audio_embeddings)
        print(filename, index + 1, "of", total)
    concat_array = np.concatenate(list_of_embeddings, axis=0)
    np.save(file=clap_npy, arr=concat_array, allow_pickle=True)


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate CLAP embeddings.")
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
        "--clap_npy",
        metavar="S",
        type=str,
        required=True,
        dest="clap_npy",
        help="Path for saving CLAP embeddings.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
