"""
Train a model using transfer learning.
"""

import argparse
import tensorflow as tf
import random

import augment
import datareader_tune
import model


# Disable GPU on TensorFlow
# TF crashes with JIT error if CUDA CUDNN not properly set up
# Stick with CPU just to be safe
tf.config.set_visible_devices(devices=[], device_type="GPU")
# Set a fixed global random seed for TF
# tf.random.set_seed(seed=1234)
tf.random.set_seed(seed=None)

# Set the fixed global random seed for module random
# random.seed(a=1234)
random.seed(a=None)

# Some hardcoded parameters used by YAMNet
desired_channels = 0
sample_rate = 16000
data_type = "int16"
stft_window_seconds = 0.025
stft_hop_seconds = 0.010
patch_window_seconds = 0.96
# patch_window_seconds_padded is required
# as the STFT algorithm uses slightly more data than
# patch_window_seconds when dividing the waveform into windows
patch_window_seconds_padded = 0.975
patch_hop_seconds = 0.48
mel_bands = 64
mel_min_hz = 125.0
mel_max_hz = 7500.0
log_offset = 0.001

# The input length for the model
# Use None for no fixed length otherwise specify length
input_length = None

# Values for marking silence
# Silence threshold. If None then no silence detection is performed.
sil_threshold = None
# Percentage of values which must exceed threshold
# for sequence to be marked as not silence
# Range [>= 0, <= 100]
sil_percentage = 40
# Onehot encoded label to represent silence
sil_vector = [0.0] * 521
sil_vector[494] = 1.0
sil_vector = tf.convert_to_tensor(value=sil_vector, dtype="float32")

# Standardisation parameters
# If either are None then normalise to -1, 1
std_mean = None
std_sd = None

# Augmentation parameters
# Shuffle data order after each training epoch
shuffle = True
# Augmentation functions
augmentations = [
    augment.augment_gain,
    augment.augment_uniform_noise,
    augment.augment_stretch,
]
# Do not shuffle or augment validation data
val_shuffle = False
val_augment = None

# Training parameters
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
loss = tf.keras.losses.CategoricalFocalCrossentropy(label_smoothing=0.1)
metrics = [tf.keras.metrics.CategoricalAccuracy()]

# Callback parameters
# Save after each epoch if val_loss improves
checkpoint_monitor = "val_loss"
checkpoint_save_best_only = True
checkpoint_save_freq = "epoch"
# Stop training if val_loss does not improve
restore_best_weights = True
stopping_patience = 6
stopping_monitor = "val_loss"
# Reduce learning rate if val_loss does not improve
reduce_monitor = "val_loss"
reduce_factor = 0.5
reduce_patience = 3


def _main(
    log_directory,
    train_filelist,
    val_filelist,
    existing_model_file,
    new_model_file,
    epochs,
):
    """Train a model using transfer learning.

    Args:
        log_directory: Log directory.
        train_filelist: A CSV containing training files and labels.
        val_filelist: A CSV containing validation files and labels.
        existing_model_file: Existing model filepath.
        new_model_file: New model filepath.
        epochs: Number of epochs.
    """
    # Prepare training data
    training_data = datareader_tune.DataReaderTune(
        filelist=train_filelist,
        desired_channels=desired_channels,
        sample_rate=sample_rate,
        data_type=data_type,
        stft_window_seconds=stft_window_seconds,
        stft_hop_seconds=stft_hop_seconds,
        patch_window_seconds=patch_window_seconds,
        patch_window_seconds_padded=patch_window_seconds_padded,
        patch_hop_seconds=patch_hop_seconds,
        mel_bands=mel_bands,
        mel_min_hz=mel_min_hz,
        mel_max_hz=mel_max_hz,
        log_offset=log_offset,
        sil_threshold=sil_threshold,
        sil_percentage=sil_percentage,
        sil_vector=sil_vector,
        std_mean=std_mean,
        std_sd=std_sd,
        shuffle=shuffle,
        augmentations=augmentations,
    )
    # Prepare validation data
    val_data = datareader_tune.DataReaderTune(
        filelist=val_filelist,
        desired_channels=desired_channels,
        sample_rate=sample_rate,
        data_type=data_type,
        stft_window_seconds=stft_window_seconds,
        stft_hop_seconds=stft_hop_seconds,
        patch_window_seconds=patch_window_seconds,
        patch_window_seconds_padded=patch_window_seconds_padded,
        patch_hop_seconds=patch_hop_seconds,
        mel_bands=mel_bands,
        mel_min_hz=mel_min_hz,
        mel_max_hz=mel_max_hz,
        log_offset=log_offset,
        sil_threshold=sil_threshold,
        sil_percentage=sil_percentage,
        sil_vector=sil_vector,
        std_mean=std_mean,
        std_sd=std_sd,
        shuffle=val_shuffle,
        augmentations=val_augment,
    )
    # Initialise model
    yamnet_model = model.build_tunable_model(
        input_length=input_length,
        existing_model_file=existing_model_file,
        num_classes=len(training_data.get_onehot()),
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    # Train model
    callback_functions = [
        tf.keras.callbacks.ModelCheckpoint(
            # filepath must have a keras extension
            filepath=new_model_file,
            monitor=checkpoint_monitor,
            save_best_only=checkpoint_save_best_only,
            save_freq=checkpoint_save_freq,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=stopping_monitor,
            patience=stopping_patience,
            restore_best_weights=restore_best_weights,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=reduce_monitor,
            factor=reduce_factor,
            patience=reduce_patience,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_directory),
    ]
    yamnet_model.fit(
        x=training_data,
        validation_data=val_data,
        callbacks=callback_functions,
        epochs=epochs,
        class_weight=None,
    )
    # Save model as keras/h5 format
    # yamnet_model.save(filepath="final_model.keras")
    # Save model as SavedModel
    # yamnet_model.export(filepath="final_model_savedmodel")


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a model using transfer learning."
    )
    parser.add_argument(
        "--log_directory",
        metavar="S",
        type=str,
        required=True,
        dest="log_directory",
        help="Log directory.",
    )
    parser.add_argument(
        "--train_filelist",
        metavar="S",
        type=str,
        required=True,
        dest="train_filelist",
        help="A CSV containing training files and labels.",
    )
    parser.add_argument(
        "--val_filelist",
        metavar="S",
        type=str,
        required=True,
        dest="val_filelist",
        help="A CSV containing validation files and labels.",
    )
    parser.add_argument(
        "--existing_model_file",
        metavar="S",
        type=str,
        required=True,
        dest="existing_model_file",
        help="Existing model filepath.",
    )
    parser.add_argument(
        "--new_model_file",
        metavar="S",
        type=str,
        required=True,
        dest="new_model_file",
        help="New model filepath.",
    )
    parser.add_argument(
        "--epochs",
        metavar="N",
        type=int,
        required=True,
        dest="epochs",
        help="Number of epochs.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
