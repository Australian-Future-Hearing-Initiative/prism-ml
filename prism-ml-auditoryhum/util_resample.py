"""
Resample audio.
"""

import argparse
import glob
import multiprocessing
import subprocess


def _resample_helper(index, total, sampling_rate, bits, channels, filename):
    """Resample audio helper.

    Args:
        index: Position of current file.
        total: Total files to be resampled.
        sampling_rate: Desired sampling rate.
        bits: Bits per sample for FFmpeg. E.g. "s16" for 16-bit signed.
        channels: Desired channels.
        filename: The filename to be resampled.
    """
    temp_filename = filename[0:-4] + "_temp.wav"
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
        str(bits),
        temp_filename,
    ]
    subprocess.call(args=convert_command)
    # Delete old file
    subprocess.call(args=["rm", filename])
    # Rename temp file
    subprocess.call(args=["mv", temp_filename, filename])
    print(filename, index + 1, total)


def _main(filedir, sampling_rate, bits, channels):
    """Resample audio.

    Args:
        filedir: Regex file path of audio. E.g. "*.wav".
        sampling_rate: Desired sampling rate.
        bits: Bits per sample for FFmpeg. E.g. "s16" for 16-bit signed.
        channels: Desired channels.
    """
    # Resampling parameters
    filelist = glob.glob(pathname=filedir)
    total = len(filelist)
    index = range(total)
    filetuple = list(
        zip(
            index,
            [total] * total,
            [sampling_rate] * total,
            [bits] * total,
            [channels] * total,
            filelist,
        )
    )
    # 4 Threads
    pool_function = multiprocessing.Pool(4)
    # Multithreaded
    pool_function.starmap(_resample_helper, filetuple)


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="Resample audio.")
    parser.add_argument(
        "--filedir",
        metavar="S",
        type=str,
        required=True,
        dest="filedir",
        help='Regex file path of audio. E.g. "*.wav".',
    )
    parser.add_argument(
        "--sampling_rate",
        metavar="N",
        type=int,
        required=True,
        dest="sampling_rate",
        help="Desired sampling rate.",
    )
    parser.add_argument(
        "--bits",
        metavar="S",
        type=str,
        required=True,
        dest="bits",
        help='Bits per sample for FFmpeg. E.g. "s16" \
                        for 16-bit signed.',
    )
    parser.add_argument(
        "--channels",
        metavar="N",
        type=int,
        required=True,
        dest="channels",
        help="Desired channels.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
