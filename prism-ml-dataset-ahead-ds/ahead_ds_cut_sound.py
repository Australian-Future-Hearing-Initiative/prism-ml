"""
Cut sound to desired lengths.
"""

import glob
import numpy as np
import soundfile as sf
import sys


def _cut_sound(input_regex, output_regex, start, stop, step):
    """Cut sound to desired lengths.

    Args:
        input_regex: Regular expression to find input audio files.
        output_regex: Regular expression for naming output audio files.
        start: Starting position for cutting audio.
        stop: Stopping position for cutting audio.
        step: Length of each audio cut.
    """
    filelist = glob.glob(pathname=input_regex)
    sequence = np.arange(start=start, stop=stop, step=step)
    findex = 1
    # Loop through each audio file
    for filename in filelist:
        waveform, sr = sf.read(file=filename, always_2d=True, dtype="int16")
        # Extract parts of the audio file
        for seq_index in sequence:
            sub_waveform = waveform[seq_index : (seq_index + step)]
            sf.write(
                file=output_regex.format(findex),
                data=sub_waveform,
                samplerate=sr,
            )
            print(filename, output_regex.format(findex))
            findex += 1


if __name__ == "__main__":
    input_regex_arg = sys.argv[1]
    output_regex_arg = sys.argv[2]
    start_arg = int(sys.argv[3])
    stop_arg = int(sys.argv[4])
    step_arg = int(sys.argv[5])
    _cut_sound(
        input_regex=input_regex_arg,
        output_regex=output_regex_arg,
        start=start_arg,
        stop=stop_arg,
        step=step_arg,
    )
