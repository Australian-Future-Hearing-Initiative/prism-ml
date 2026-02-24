"""
Calculate MFCC.
"""

import argparse
import glob
import librosa
import numpy as np


# Signal processing parameters
n_fft = 2048
hop_length = 512
win_length = 2048
n_mfcc = 20
sr = 16000


def _main(filedir, mfcc_npy):
    """Calculate MFCC.

    Args:
        filedir: Regex file path of audio. E.g. "*.wav".
        mfcc_npy: Path for saving MFCC.
    """
    filelist = glob.glob(pathname=filedir)
    filelist.sort()
    total = len(filelist)
    list_of_embeddings = list()
    for index, filename in enumerate(filelist):
        y, _ = librosa.load(path=filename, sr=sr, mono=True)
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mfcc=n_mfcc,
        )
        mean_mfcc = np.mean(a=mfccs.T, axis=0)
        list_of_embeddings.append(mean_mfcc)
        print(filename, index + 1, "of", total)
    concat_array = np.stack(arrays=list_of_embeddings, axis=0)
    np.save(file=mfcc_npy, arr=concat_array, allow_pickle=True)


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate MFCC.")
    parser.add_argument(
        "--filedir",
        metavar="S",
        type=str,
        required=True,
        dest="filedir",
        help='Regex file path of audio. E.g. "*.wav".',
    )
    parser.add_argument(
        "--mfcc_npy",
        metavar="S",
        type=str,
        required=True,
        dest="mfcc_npy",
        help="Path for saving MFCC.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
