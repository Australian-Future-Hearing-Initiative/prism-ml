"""
Resample audio to desired sampling rate.
"""

import glob
import multiprocessing
import subprocess
import sys


def _resample_helper(index, total, sampling_rate, channels, filename):
    """Resample audio helper.

    Args:
        index: Position of current file.
        total: Total files to be resampled.
        sampling_rate: Desired sampling rate.
        channels: Desired channels.
        filename: The filename to be resampled.
    """
    temp_filename = filename + "_temp.wav"
    convert_command = [
        "ffmpeg",
        "-y",
        "-i",
        filename,
        "-vn",
        "-ar",
        str(sampling_rate),
        "-ac",
        str(channels),
        "-sample_fmt",
        "s16",
        temp_filename,
    ]
    subprocess.call(args=convert_command)
    # Delete old file
    subprocess.call(args=["rm", filename])
    # Rename temp file
    subprocess.call(args=["mv", temp_filename, filename])
    print(filename, index + 1, total)


def _resample_audio(filedir, sampling_rate, channels):
    """Resample audio to desired sampling rate.

    Args:
        filedir: Directory containing audio.
        sampling_rate: Desired sampling rate.
        channels: Desired channels.
    """
    # Resampling parameters
    filelist = glob.glob(pathname=filedir)
    total = len(filelist)
    index = range(total)
    param_list = list(
        zip(
            index,
            [total] * total,
            [sampling_rate] * total,
            [channels] * total,
            filelist,
        )
    )
    # 4 Threads
    pool_function = multiprocessing.Pool(4)
    # Multithreaded
    pool_function.starmap(_resample_helper, param_list)


if __name__ == "__main__":
    filedir_arg = sys.argv[1]
    sampling_rate_arg = sys.argv[2]
    channels_arg = sys.argv[3]
    _resample_audio(
        filedir=filedir_arg,
        sampling_rate=sampling_rate_arg,
        channels=channels_arg,
    )
