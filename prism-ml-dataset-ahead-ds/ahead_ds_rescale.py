"""
Rescale sound to desired volume.
"""

import glob
import numpy as np
import multiprocessing
import soundfile as sf
import sys


def _rescale_sound_helper(filename, old_rms, new_rms):
    """Rescale sound to desired volume helper.

    Args:
        filename: The filename to be rescaled.
        old_rms: Old RMS.
        new_rms: New RMS.
    """
    waveform, sr = sf.read(file=filename, always_2d=True, dtype="int16")
    scale = new_rms / old_rms
    waveform = waveform * scale
    info_int16 = np.iinfo(int_type=np.int16)
    min_val = info_int16.min
    max_val = info_int16.max
    waveform = np.clip(a=waveform, a_min=min_val, a_max=max_val)
    waveform = waveform.astype(dtype=np.int16)
    sf.write(file=filename, data=waveform, samplerate=sr)
    print(filename)


def _rescale_sound(input_regex, old_rms, new_rms):
    """Rescale sound to desired volume.

    Args:
        input_regex: Regular expression to find input audio files.
        old_rms: Old RMS.
        new_rms: New RMS.
    """
    file_list = glob.glob(input_regex)
    total = len(file_list)
    param_list = list(zip(file_list, [old_rms] * total, [new_rms] * total))
    # 4 Threads
    pool_function = multiprocessing.Pool(4)
    # Multithreaded
    pool_function.starmap(_rescale_sound_helper, param_list)


if __name__ == "__main__":
    input_regex_arg = sys.argv[1]
    old_rms_arg = float(sys.argv[2])
    new_rms_arg = float(sys.argv[3])
    _rescale_sound(
        input_regex=input_regex_arg, old_rms=old_rms_arg, new_rms=new_rms_arg
    )
