"""
Move and rename speech files for later mixing.
"""

import glob
import itertools
import multiprocessing
import subprocess
import sys


def _split_speech_helper(source_file, dest_file):
    """Move and rename speech files helper.

    Args:
        source_file: Source file.
        dest_file: Destination file.
    """
    subprocess.call(args=["mv", source_file, dest_file])
    print(source_file, dest_file)


def _split_speech(
    speech_regex,
    environ1_regex,
    environ2_regex,
    environ3_regex,
    environ4_regex,
    environ5_regex,
    environ6_regex,
    speech1_regex,
    speech2_regex,
    speech3_regex,
    speech4_regex,
    speech5_regex,
    speech6_regex,
    remaining_speech_regex,
):
    """Move and rename speech files for later mixing.

    Args:
        speech_regex: Regex representing the speech source audio.
        environ1_regex: Regex representing the name of an
        environment audio class 1.
        environ2_regex: Regex representing the name of an
        environment audio class 2.
        environ3_regex: Regex representing the name of an
        environment audio class 3.
        environ4_regex: Regex representing the name of an
        environment audio class 4.
        environ5_regex: Regex representing the name of an
        environment audio class 5.
        environ6_regex: Regex representing the name of an
        environment audio class 6.
        speech1_regex: Regex for directories of speech to be
        associated with environ1.
        speech2_regex: Regex for directories of speech to be
        associated with environ2.
        speech3_regex: Regex for directories of speech to be
        associated with environ3.
        speech4_regex: Regex for directories of speech to be
        associated with environ4.
        speech5_regex: Regex for directories of speech to be
        associated with environ5.
        speech6_regex: Regex for directories of speech to be
        associated with environ6.
        remaining_speech_regex: Regex for directories of
        remaining speech.
    """
    speech_list = glob.glob(pathname=speech_regex)
    environ1_list = glob.glob(pathname=environ1_regex)
    environ2_list = glob.glob(pathname=environ2_regex)
    environ3_list = glob.glob(pathname=environ3_regex)
    environ4_list = glob.glob(pathname=environ4_regex)
    environ5_list = glob.glob(pathname=environ5_regex)
    environ6_list = glob.glob(pathname=environ6_regex)
    # Get the number of existing speech and environment audio files
    speech_length = len(speech_list)
    environ1_length = len(environ1_list)
    environ2_length = len(environ2_list)
    environ3_length = len(environ3_list)
    environ4_length = len(environ4_list)
    environ5_length = len(environ5_list)
    environ6_length = len(environ6_list)
    existing_length = (
        environ1_length
        + environ2_length
        + environ3_length
        + environ4_length
        + environ5_length
        + environ6_length
    )
    remaining_length = speech_length - existing_length
    # Assign names for new audio files
    speech1_list = [speech1_regex.format(x + 1) for x in range(environ1_length)]
    speech2_list = [speech2_regex.format(x + 1) for x in range(environ2_length)]
    speech3_list = [speech3_regex.format(x + 1) for x in range(environ3_length)]
    speech4_list = [speech4_regex.format(x + 1) for x in range(environ4_length)]
    speech5_list = [speech5_regex.format(x + 1) for x in range(environ5_length)]
    speech6_list = [speech6_regex.format(x + 1) for x in range(environ6_length)]
    remaining_list = [
        remaining_speech_regex.format(x + 1) for x in range(remaining_length)
    ]
    # Interleave the file names
    chained = itertools.zip_longest(
        speech1_list,
        speech2_list,
        speech3_list,
        speech4_list,
        speech5_list,
        speech6_list,
        remaining_list,
    )
    dest_interleave = [
        x for x in itertools.chain.from_iterable(chained) if x is not None
    ]
    param_list = list(zip(speech_list, dest_interleave))
    # 4 Threads
    pool_function = multiprocessing.Pool(4)
    # Multithreaded
    pool_function.starmap(_split_speech_helper, param_list)


if __name__ == "__main__":
    speech_regex_arg = sys.argv[1]
    environ1_regex_arg = sys.argv[2]
    environ2_regex_arg = sys.argv[3]
    environ3_regex_arg = sys.argv[4]
    environ4_regex_arg = sys.argv[5]
    environ5_regex_arg = sys.argv[6]
    environ6_regex_arg = sys.argv[7]
    speech1_regex_arg = sys.argv[8]
    speech2_regex_arg = sys.argv[9]
    speech3_regex_arg = sys.argv[10]
    speech4_regex_arg = sys.argv[11]
    speech5_regex_arg = sys.argv[12]
    speech6_regex_arg = sys.argv[13]
    remaining_speech_regex_arg = sys.argv[14]
    _split_speech(
        speech_regex=speech_regex_arg,
        environ1_regex=environ1_regex_arg,
        environ2_regex=environ2_regex_arg,
        environ3_regex=environ3_regex_arg,
        environ4_regex=environ4_regex_arg,
        environ5_regex=environ5_regex_arg,
        environ6_regex=environ6_regex_arg,
        speech1_regex=speech1_regex_arg,
        speech2_regex=speech2_regex_arg,
        speech3_regex=speech3_regex_arg,
        speech4_regex=speech4_regex_arg,
        speech5_regex=speech5_regex_arg,
        speech6_regex=speech6_regex_arg,
        remaining_speech_regex=remaining_speech_regex_arg,
    )
