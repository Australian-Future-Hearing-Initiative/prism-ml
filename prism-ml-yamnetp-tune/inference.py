"""
Perform sound recognition.
"""

import argparse
import tensorflow as tf

import datareader_tune
import train_transfer


# Disable GPU on TensorFlow
# TF crashes with JIT error if CUDA CUDNN not properly set up
# Stick with CPU just to be safe
tf.config.set_visible_devices(devices=[], device_type="GPU")


def _main_tf(model_file, sound_file):
    """Perform sound recognition.

    Args:
        model_file: Model filepath.
        sound_file: Sound filepath.
    """
    # Load waveform
    waveform = datareader_tune.DataReaderTune.load_sound_tf(
        filename=sound_file,
        desired_channels=train_transfer.desired_channels,
        sample_rate=train_transfer.sample_rate,
        data_type=train_transfer.data_type,
        patch_window_seconds=train_transfer.patch_window_seconds,
        std_mean=train_transfer.std_mean,
        std_sd=train_transfer.std_sd,
    )
    # Load model
    model = tf.keras.models.load_model(filepath=model_file)
    # Inference
    scores = model(waveform)
    max_score = tf.math.reduce_max(input_tensor=scores, axis=1)
    class_index = tf.math.argmax(input=scores, axis=1)
    print("Max score", max_score)
    print("Class index", class_index)


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="Perform sound recognition.")
    parser.add_argument(
        "--model_file",
        metavar="S",
        type=str,
        required=True,
        dest="model_file",
        help="Model filepath.",
    )
    parser.add_argument(
        "--sound_file",
        metavar="S",
        type=str,
        required=True,
        dest="sound_file",
        help="Sound filepath.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main_tf(**arguments)
