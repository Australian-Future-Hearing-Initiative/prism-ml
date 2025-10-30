"""
LiteRT conversion.
"""

import argparse
import tensorflow as tf


def _main(keras_file, liteRT_file):
    """LiteRT conversion.

    Args:
        keras_file: Keras model filepath.
        liteRT_file: LiteRT model filepath.
    """
    # Create LiteRT converter from Keras model
    yamnet_model = tf.keras.models.load_model(filepath=keras_file)
    converter = tf.lite.TFLiteConverter.from_keras_model(model=yamnet_model)
    # Perform quantisation
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Select the data type for quantisation, comment out to stay with float32
    # In testing float16 produces 1/2 size and retains good results
    converter.target_spec.supported_types = [tf.float16]

    # Convert model
    liteRT_model = converter.convert()
    f = open(file=liteRT_file, mode="wb")
    f.write(liteRT_model)
    f.close()


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="LiteRT conversion.")
    parser.add_argument(
        "--keras_file",
        metavar="S",
        type=str,
        required=True,
        dest="keras_file",
        help="Keras model filepath.",
    )
    parser.add_argument(
        "--liteRT_file",
        metavar="S",
        type=str,
        required=True,
        dest="liteRT_file",
        help="LiteRT model filepath.",
    )
    arguments = vars(parser.parse_args())
    return arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
