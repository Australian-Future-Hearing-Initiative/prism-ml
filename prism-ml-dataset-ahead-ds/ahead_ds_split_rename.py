"""
Move and rename files.
"""

import glob
import itertools
import math
import multiprocessing
import subprocess
import sys


def _move_rename_helper(source_file, dest_file):
    """Move and rename files helper.

    Args:
        source_file: Source file.
        dest_file: Destination file.
    """
    subprocess.call(args=["mv", source_file, dest_file])
    print(source_file, dest_file)


def _split_rename(source_regex, dest1_regex, dest2_regex):
    """Move and rename files.

    Args:
        source_regex: Regex representing the name of source audio.
        dest1_regex: Regex representing the name of first dest audio.
        dest2_regex: Regex representing the name of second dest audio.
    """
    source_list = glob.glob(pathname=source_regex)
    total = len(source_list)
    dest1_names = [
        dest1_regex.format(x + 1) for x in range(math.ceil(total / 2))
    ]
    dest2_names = [
        dest2_regex.format(x + 1) for x in range(math.floor(total / 2))
    ]
    dest_interleave = [
        x
        for x in itertools.chain.from_iterable(
            itertools.zip_longest(dest1_names, dest2_names)
        )
        if x is not None
    ]
    param_list = list(zip(source_list, dest_interleave))
    # 4 Threads
    pool_function = multiprocessing.Pool(4)
    # Multithreaded
    pool_function.starmap(_move_rename_helper, param_list)


if __name__ == "__main__":
    source_regex_arg = sys.argv[1]
    dest1_regex_arg = sys.argv[2]
    dest2_regex_arg = sys.argv[3]
    _split_rename(
        source_regex=source_regex_arg,
        dest1_regex=dest1_regex_arg,
        dest2_regex=dest2_regex_arg,
    )
