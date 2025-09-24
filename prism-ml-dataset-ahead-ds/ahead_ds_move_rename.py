"""
Move and rename files.
"""

import glob
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


def _move_rename(source_regex, dest_regex):
    """Move and rename files.

    Args:
        source_regex: Regex representing the name of source audio.
        dest_regex: Regex representing the name of dest audio.
    """
    source_list = glob.glob(pathname=source_regex)
    total = len(source_list)
    dest_list = [dest_regex.format((x + 1)) for x in range(total)]
    param_list = list(zip(source_list, dest_list))
    # 4 Threads
    pool_function = multiprocessing.Pool(4)
    # Multithreaded
    pool_function.starmap(_move_rename_helper, param_list)


if __name__ == "__main__":
    source_regex_arg = sys.argv[1]
    dest_regex_arg = sys.argv[2]
    _move_rename(source_regex=source_regex_arg, dest_regex=dest_regex_arg)
