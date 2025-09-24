"""
Analyse sounds to find RMS.
"""

import glob
import numpy as np
import soundfile as sf
import sys


def _analyse_sound(input_regex):
    """Analyse sounds to find RMS.

    Args:
        input_regex: Regular expressions representing input
        audio files.
    """
    filelist = glob.glob(pathname=input_regex)
    set_sum = list()
    set_count = list()
    for filename in filelist:
        waveform, _ = sf.read(file=filename, always_2d=True, dtype="int16")
        waveform = waveform.astype(dtype=np.float32)
        summed = np.sum(a=np.square(waveform))
        set_sum.append(summed)
        set_count.append(waveform.shape[0])
        print(filename)
    total = np.sum(a=set_sum)
    count = np.sum(a=set_count)
    root_ms = np.sqrt(total / count)
    print("Root mean squared", root_ms)


if __name__ == "__main__":
    input_regex_args = sys.argv[1]
    _analyse_sound(input_regex=input_regex_args)
