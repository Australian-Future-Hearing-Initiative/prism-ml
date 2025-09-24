"""
This module contains model definitions.
"""

import tensorflow as tf


def build_tunable_model(
    existing_model_file, input_length, num_classes, optimizer, loss, metrics
):
    """Create a model used for transfer learning based on YAMNet.

    Args:
        existing_model_file: Directory of existing YAMNet model file.
        input_length: Input length of the model. None for no fixed length.
        num_classes: Number of classes.
        optimizer: Optimizer.
        loss: Loss function.
        metrics: Metrics during training.

    Returns:
        Tunable YAMNet model.
    """
    # Load and modify model for tuning
    if input_length:
        input_layer = tf.keras.Input(shape=(input_length,))
    else:
        input_layer = tf.keras.Input(shape=())
    existing_model = tf.keras.layers.TFSMLayer(
        filepath=existing_model_file, call_endpoint="serving_default"
    )
    layer01 = existing_model(input_layer)["output_1"]
    output = tf.keras.layers.Dense(units=num_classes, activation="sigmoid")(
        layer01
    )
    model = tf.keras.Model(inputs=input_layer, outputs=output)

    # Select frozen weights here
    for weight in model.weights:
        # TODO - bug
        # TFSMLayer might be missing setter trainable
        # Instead use _trainable
        weight._trainable = True
    # Last 2 layers are the dense layer
    model.weights[-1].trainable = True
    model.weights[-2].trainable = True

    # Set regularizer
    for weight in model.weights:
        # TODO - bug
        # TF weights have no regularizer
        # This causes Keras to crash
        # Set a regularizer attribute
        if not hasattr(weight, "regularizer"):
            setattr(weight, "regularizer", None)

    # Set path
    for weight in model.weights:
        # TODO - bug
        # tensorflow/python/framework/ops.py contains value
        # _VALID_SCOPE_NAME_REGEX, All layer path names must
        # conform to the regular expression set in
        # _VALID_SCOPE_NAME_REGEX
        # For now just replace ":"
        weight._path = weight._path.replace(":", "_")

    # Compile model
    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False
    )
    return model
