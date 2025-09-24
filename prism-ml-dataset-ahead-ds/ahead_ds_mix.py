"""
Mix sounds to desired dBfs SNR.
"""

import glob
import itertools
import numpy as np
import multiprocessing
import soundfile as sf
import sys


def _rms(waveform):
    """Find RMS.

    Args:
        waveform: Input waveform. Range [-1.0, 1.0].

    Returns:
        RMS of waveform.
    """
    mean_squared = np.mean(a=np.square(waveform))
    rms = np.sqrt(mean_squared)
    return rms


def _mix_signal(signal, noise, snr):
    """Mix signal and noise to desired SNR.

    Args:
        signal: Signal waveform.
        noise: Noise waveform.
        snr: SNR value in dBfs.

    Returns:
        Mixed signal and noise waveform.
    """
    # Get the RMS
    signal_rms = _rms(waveform=signal)
    noise_rms = _rms(waveform=noise)
    # Ensure no divide by zero errors
    signal_rms = signal_rms if signal_rms > 0.0 else 1e-6
    noise_rms = noise_rms if noise_rms > 0.0 else 1e-6
    # Boost the quieter sound
    if signal_rms > noise_rms:
        signal_boosted = signal
        noise_boosted = noise * signal_rms / noise_rms
    else:
        signal_boosted = signal * noise_rms / signal_rms
        noise_boosted = noise
    # Set the gain for signal
    gain_factor = 10.0 ** (snr / 20.0)
    signal_gained = signal_boosted * gain_factor
    # Final sound after mixing
    new_sound = signal_gained + noise_boosted
    return new_sound


def _mix_helper(speech_file, environ_file, new_file, snr):
    """Mix sounds to desired SNR helper.

    Args:
        speech_file: The filename of speech sounds.
        environ_file: The filename of environment sounds.
        new_file: Filename of output.
        snr: SNR value in dB.
    """
    # Speech sounds
    speech, sr = sf.read(file=speech_file, always_2d=True, dtype="int16")
    speech = speech.astype(dtype=np.float32)
    # Environment sounds
    environ, sr = sf.read(file=environ_file, always_2d=True, dtype="int16")
    environ = environ.astype(dtype=np.float32)
    # Mix sounds
    new_waveform = _mix_signal(signal=speech, noise=environ, snr=snr)
    # Convert and save sounds
    info_int16 = np.iinfo(int_type=np.int16)
    min_val = info_int16.min
    max_val = info_int16.max
    new_waveform = np.clip(a=new_waveform, a_min=min_val, a_max=max_val)
    new_waveform = new_waveform.astype(dtype=np.int16)
    sf.write(file=new_file, data=new_waveform, samplerate=sr)
    print(environ_file, speech_file, new_file, snr)


def _mix(environ_regex, speech_regex, output_regex, snr):
    """Mix sounds to desired dBfs SNR.

    Args:
        environ_regex: Regex for environment sounds.
        speech_regex: Regex for speech sounds.
        output_regex: Regex representing output.
        snr: List of SNR values in dB.
    """
    environ_list = glob.glob(pathname=environ_regex)
    speech_list = glob.glob(pathname=speech_regex)
    total = len(environ_list)
    output_list = [output_regex.format(x + 1) for x in range(total)]
    snr_part = [int(x) for x in snr.split(",")]
    snr_list = list(itertools.islice(itertools.cycle(snr_part), total))
    param_list = list(zip(speech_list, environ_list, output_list, snr_list))
    # 4 Threads
    pool_function = multiprocessing.Pool(4)
    # Multithreaded
    pool_function.starmap(_mix_helper, param_list)


if __name__ == "__main__":
    environ_regex_arg = sys.argv[1]
    speech_regex_arg = sys.argv[2]
    output_regex_arg = sys.argv[3]
    snr_arg = sys.argv[4]
    _mix(
        environ_regex=environ_regex_arg,
        speech_regex=speech_regex_arg,
        output_regex=output_regex_arg,
        snr=snr_arg,
    )
