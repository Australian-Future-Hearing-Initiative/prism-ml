"""
This module contains augmentation code.
"""

import random
import tensorflow as tf


def augment_gain(waveform, max_db=6, min_db=-6, prob=0.5):
    """Augment by applying gain.

    Args:
        waveform: Input waveform. Range [-1.0, 1.0].
        max_db: Maximum gain in db. Range >= min_db.
        min_db: Minimum gain in db.
        prob: Probability to apply augmentation. Range [0.0, 1.0].

    Returns:
        Augmented waveform.
    """
    event_prob = random.random()
    if prob > event_prob:
        rand_gain_db = random.randint(a=min_db, b=max_db)
        waveform = waveform * 10 ** (rand_gain_db / 20)
        waveform = tf.clip_by_value(
            t=waveform, clip_value_min=-1.0, clip_value_max=1.0
        )
    return waveform


def _rms(waveform):
    """Find RMS.

    Args:
        waveform: Input waveform. Range [-1.0, 1.0].

    Returns:
        RMS of waveform.
    """
    squared = tf.math.square(x=waveform)
    mean_squared = tf.math.reduce_mean(input_tensor=squared)
    rms = tf.math.sqrt(x=mean_squared)
    return rms


def augment_uniform_noise(
    waveform, max_n=0.003, min_n=-0.003, seed=123, prob=0.5
):
    """Augment by applying uniform noise.

    Args:
        waveform: Input waveform. Range [-1.0, 1.0].
        max_n: Maximum value for noise. Range >= min_mag.
        min_n: Minimum value for noise. Range [-1.0, 1.0].
        seed: Random seed.
        prob: Probability to apply augmnentation. Range [0.0, 1.0].

    Returns:
        Augmented waveform.
    """
    event_prob = random.random()
    if prob > event_prob:
        # Create noise
        gen = tf.random.Generator.from_seed(seed=seed)
        noise = gen.uniform(shape=waveform.shape, minval=min_n, maxval=max_n)
        # Add noise
        waveform = waveform + noise
        waveform = tf.clip_by_value(
            t=waveform, clip_value_min=-1.0, clip_value_max=1.0
        )
    return waveform


def _resample_fourier_tf(x, num_samples):
    """
    Resamples a 1D tensor x to num_samples using the
    Fourier method in TensorFlow.

    Args:
        x: A 1D tf.Tensor representing the input signal.
        num_samples: An integer, the desired number of output samples.

    Returns:
        A 1D tf.Tensor representing the resampled signal.
    """
    x_len = tf.shape(x)[0]
    dtype = x.dtype

    # 1. Forward FFT
    # tf.signal.fft takes complex inputs, so convert x to complex.
    x_complex = tf.cast(x=x, dtype=tf.complex64)
    x_fft = tf.signal.fft(input=x_complex)

    # 2. Pad/Truncate in Frequency Domain
    if num_samples == x_len:
        # No resampling needed
        return x

    if num_samples > x_len:
        # Upsampling: Pad with zeros in the middle of the frequency spectrum
        # We need to handle even/odd lengths for both input and output correctly
        if x_len % 2 == 0:
            # Even length input: DC component at x_fft[0],
            # Nyquist at x_fft[x_len // 2]
            positive_freqs = x_fft[1 : x_len // 2]
            negative_freqs = x_fft[
                x_len // 2 + 1 :
            ]  # Includes Nyquist if present, and higher negative freqs
            nyquist = x_fft[
                x_len // 2 : x_len // 2 + 1
            ]  # Nyquist frequency (if even)
            dc = x_fft[0:1]

            # Calculate padding needed
            padding_len = num_samples - x_len
            zeros_padding = tf.zeros(shape=padding_len, dtype=tf.complex64)

            # Reconstruct the padded spectrum
            if num_samples % 2 == 0:  # Even output length
                # For even output length, Nyquist will be at num_samples // 2
                padded_fft = tf.concat(
                    values=[
                        dc,
                        positive_freqs,
                        nyquist,
                        zeros_padding,
                        negative_freqs,
                    ],
                    axis=0,
                )
            else:  # Odd output length
                # For odd output length, there's no explicit Nyquist component,
                # it's absorbed into the last positive frequency component due
                # to the definition of DFT.
                # So we drop the nyquist component from the input and
                # distribute its energy if needed, or just pad with zeros
                # directly without explicitly handling it in this
                # simple scheme.
                # For simplicity here, if output is odd and input
                # was even, we might just drop nyquist and pad.
                # A more rigorous implementation would distribute
                # the Nyquist component.
                # For now, let's just pad differently.
                # It's safer to remove the nyquist when going from even
                # to odd output.
                padded_fft = tf.concat(
                    values=[dc, positive_freqs, zeros_padding, negative_freqs],
                    axis=0,
                )
        else:
            # Odd length input: DC component at x_fft[0]
            positive_freqs = x_fft[1 : (x_len + 1) // 2]
            negative_freqs = x_fft[(x_len + 1) // 2 :]
            dc = x_fft[0:1]
            # Calculate padding needed
            padding_len = num_samples - x_len
            zeros_padding = tf.zeros(shape=padding_len, dtype=tf.complex64)
            # Reconstruct the padded spectrum
            padded_fft = tf.concat(
                values=[dc, positive_freqs, zeros_padding, negative_freqs],
                axis=0,
            )
    else:  # num_samples < x_len
        # Downsampling: Truncate high-frequency components
        # This means keeping only the central part of the frequency spectrum.
        # We need to be careful with the DC component (index 0)
        # and the Nyquist frequency.
        if x_len % 2 == 0:
            # Even length input
            # num_samples // 2 positive frequencies, num_samples // 2 - 1
            # negative frequencies
            # or num_samples // 2 for negative if num_samples is even.
            if num_samples % 2 == 0:
                # Even output length
                truncated_fft = tf.concat(
                    values=[
                        x_fft[0 : num_samples // 2],
                        x_fft[x_len - num_samples // 2 :],
                    ],
                    axis=0,
                )
            else:
                # Odd output length
                truncated_fft = tf.concat(
                    values=[
                        x_fft[0 : (num_samples + 1) // 2],
                        x_fft[x_len - (num_samples - (num_samples + 1) // 2) :],
                    ],
                    axis=0,
                )
        else:
            # Odd length input
            truncated_fft = tf.concat(
                values=[
                    x_fft[0 : (num_samples + 1) // 2],
                    x_fft[x_len - (num_samples - (num_samples + 1) // 2) :],
                ],
                axis=0,
            )
        padded_fft = truncated_fft

    # 3. Inverse FFT
    # tf.signal.ifft expects complex inputs.
    resampled_x_complex = tf.signal.ifft(input=padded_fft)
    # The result will be complex, take the real part and
    # cast back to original dtype
    resampled_x = tf.cast(
        x=tf.math.real(input=resampled_x_complex), dtype=dtype
    )
    return resampled_x


def augment_stretch(waveform, max_rate=1.1, min_rate=0.9, prob=0.5):
    """Augment by stretching or shrinking audio run time.

    Args:
        waveform: Input 1D waveform. Range [-1.0, 1.0].
        max_rate: Max stretch rate. Range >= min_rate.
        min_rate: Min stretch rate. Range > 0.
        prob: Probability to apply augmnentation. Range [0.0, 1.0].

    Returns:
        Augmented waveform.
    """
    # Only 1D tensor currently supported
    if len(waveform.shape) != 1:
        raise ValueError("Only 1D tensor currently supported")
    event_prob = random.random()
    if prob > event_prob:
        # Stretch by an amount between max_rate and min_rate
        stretch_factor = min_rate + ((max_rate - min_rate) * random.random())
        num_samples = int(waveform.shape[0] * stretch_factor)
        # Resample using FFT
        resampled_waveform = _resample_fourier_tf(
            x=waveform, num_samples=num_samples
        )
        # Keep the same length by truncating or padding
        if resampled_waveform.shape[0] > waveform.shape[0]:
            waveform = resampled_waveform[0 : waveform.shape[0]]
        else:
            waveform = tf.zeros(shape=waveform.shape, dtype=waveform.dtype)
            indices = [[x] for x in range(resampled_waveform.shape[0])]
            waveform = tf.tensor_scatter_nd_update(
                tensor=waveform, indices=indices, updates=resampled_waveform
            )
        waveform = tf.clip_by_value(
            t=waveform, clip_value_min=-1.0, clip_value_max=1.0
        )
    return waveform
