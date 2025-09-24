"""
This module contains dataloader to process
audio into format useable by the model.
"""

import csv
import random
import soundfile as sf
import tensorflow as tf


class DataReaderTune(tf.keras.utils.Sequence):
    """Given a list of sound files create a sequence
    which returns sound with label.
    """

    def __init__(
        self,
        filelist,
        desired_channels,
        sample_rate,
        data_type,
        stft_window_seconds,
        stft_hop_seconds,
        patch_window_seconds,
        patch_window_seconds_padded,
        patch_hop_seconds,
        mel_bands,
        mel_min_hz,
        mel_max_hz,
        log_offset,
        sil_threshold,
        sil_percentage,
        sil_vector,
        std_mean,
        std_sd,
        shuffle,
        augmentations,
    ):
        """Initialise sequence.

        Args:
            filelist: A CSV containing the list of files and labels.
            desired_channels: Get input from these channels.
            sample_rate: Sample rate in Hz.
            data_type: Date type of sound file.
            stft_window_seconds: STFT window.
            stft_hop_seconds: STFT step.
            patch_window_seconds: Patch window.
            patch_window_seconds_padded: Patch window with padding.
            patch_hop_seconds: Patch step.
            mel_bands: Mel bands.
            mel_min_hz: Mel min frequency.
            mel_max_hz: Mel max frequency.
            log_offset: Log Mel offset.
            sil_threshold: Silence threshold. If None then no silence
            detection is performed.
            sil_percentage: Percentage of values which must exceed threshold
            for sequence to be marked as not silence. Range [>= 0, <= 100].
            sil_vector: The onehot encoded label to represent silence.
            std_mean: Standardisation mean. If None normalise input.
            std_sd: Standardisation standard deviation. If None normalise input.
            shuffle: Shuffle data order after each epoch.
            augmentations: A List of functions used for augmentation.
        """
        super().__init__()
        # OpenYAMNet/YAMNet+/YAMNet parameters
        self.desired_channels = desired_channels
        self.sample_rate = sample_rate
        self.data_type = data_type
        self.stft_window_seconds = stft_window_seconds
        self.stft_hop_seconds = stft_hop_seconds
        self.patch_window_seconds = patch_window_seconds
        self.patch_window_seconds_padded = patch_window_seconds_padded
        self.patch_hop_seconds = patch_hop_seconds
        self.mel_bands = mel_bands
        self.mel_min_hz = mel_min_hz
        self.mel_max_hz = mel_max_hz
        self.log_offset = log_offset
        # Silence threshold parameters
        self.sil_threshold = sil_threshold
        self.sil_percentage = sil_percentage
        self.sil_vector = sil_vector
        # Standardisation parameters
        self.std_mean = std_mean
        self.std_sd = std_sd
        # Augmentation ans shuffle parameters
        self.shuffle = shuffle
        self.augmentations = augmentations
        # Initialise file list
        self.filelist = filelist
        self.dataset, self.onehot, self.class_count = (
            DataReaderTune._get_dataset_tf(filelist=self.filelist)
        )
        self.class_weights = DataReaderTune._class_weights_tf(
            class_count=self.class_count
        )

    def __len__(self):
        """Get length of data.

        Returns:
            Length of data.
        """
        length = int(len(self.dataset))
        return length

    def __getitem__(self, idx):
        """Get sound and label.

        Args:
            idx: Index of the sound file.

        Returns:
            Arrays consisting of features and ground truth label.
        """
        filename, onehot = self.dataset[idx]
        # Get data
        waveform = DataReaderTune.load_sound_tf(
            filename=filename,
            desired_channels=self.desired_channels,
            sample_rate=self.sample_rate,
            data_type=self.data_type,
            patch_window_seconds=self.patch_window_seconds,
            std_mean=self.std_mean,
            std_sd=self.std_sd,
        )
        waveform_augmented = DataReaderTune._augmentations(
            waveform=waveform, augmentations=self.augmentations
        )
        # Padd and split waveform
        waveform_padded = DataReaderTune._pad_waveform_tf(
            waveform=waveform_augmented,
            sample_rate=self.sample_rate,
            stft_window_seconds=self.stft_window_seconds,
            stft_hop_seconds=self.stft_hop_seconds,
            patch_window_seconds=self.patch_window_seconds,
            patch_hop_seconds=self.patch_hop_seconds,
        )
        waveform_windows = DataReaderTune._waveform_windows_tf(
            waveform=waveform_padded,
            sample_rate=self.sample_rate,
            patch_window_seconds_padded=self.patch_window_seconds_padded,
            patch_hop_seconds=self.patch_hop_seconds,
        )
        # Get label
        label_windows = DataReaderTune._preprocess_label_tf(
            waveform_windows=waveform_windows,
            onehot=onehot,
            sil_threshold=self.sil_threshold,
            sil_percentage=self.sil_percentage,
            sil_vector=self.sil_vector,
        )
        return waveform_augmented, label_windows

    def on_epoch_end(self):
        """Perform after each epoch."""
        if self.shuffle:
            random.shuffle(x=self.dataset)

    def get_onehot(self):
        """Get onehot encoded dictionary.

        Returns:
            Onehot encoded dictionary.
        """
        return self.onehot

    def get_class_weights(self):
        """Get class weights encoded as a dictionary.

        Returns:
            Class weights encoded as a dictionary.
        """
        return self.class_weights

    @staticmethod
    def _augmentations(waveform, augmentations):
        """Apply augmentation.

        Args:
            waveform: The normalised sound file.
            augmentations: A List of functions used for augmentation.

        Returns:
            Augmented waveform.
        """
        if augmentations:
            for function in augmentations:
                waveform = function(waveform=waveform)
        return waveform

    @staticmethod
    def _preprocess_label_tf(
        waveform_windows, onehot, sil_threshold, sil_percentage, sil_vector
    ):
        """Preprocess label by detecting and labelling silence.

        Args:
            waveform_windows: Waveform divided into windows.
            Shape is [<batch size>, <window length>].
            onehot: Onehot encoded label for the given waveform.
            sil_threshold: Silence threshold. If None then no silence
            detection is performed.
            sil_percentage: Percentage of values which must exceed threshold
            for sequence to be marked as not silence. Range [>= 0, <= 100].
            sil_vector: The onehot encoded label to represent silence.

        Returns:
            Labels with shape [<batch size>, <onehot values>].
        """
        # Assign labels to each window in batch
        label_list = [onehot] * waveform_windows.shape[0]
        # Only preprocess silence if waveform threshold provided
        if sil_threshold:
            for index in range(waveform_windows.shape[0]):
                # The absolute value of the waveform is taken as we
                # only care about the magnitude change from zero
                window_abs = tf.math.abs(x=waveform_windows[index])
                # Produce flag to indicate if sequence should be
                # marked as silence
                silence_flag = DataReaderTune._flag_threshold_tf(
                    sequence=window_abs,
                    threshold=sil_threshold,
                    percentage=sil_percentage,
                )
                if silence_flag:
                    label_list[index] = sil_vector
        label = tf.stack(values=label_list, axis=0)
        return label

    @staticmethod
    def _flag_threshold_tf(sequence, threshold, percentage):
        """Produce flag to indicate if sequence should be flagged.
        This function was developed as calculating a percentile from sequence
        was too slow.

        Args:
            sequence: Sequence of values.
            threshold: Flagged threshold.
            percentage: Percentage of values which must exceed threshold
            for sequence to be marked as not silence. Range [> 0, < 100].

        Returns:
            Flag to indicate if sequence should be marked as silence.
        """
        total = tf.cast(x=sequence.shape[0], dtype="float32")
        exceeded = tf.reduce_sum(
            tf.cast(x=sequence > threshold, dtype="float32")
        )
        percentage_tf = tf.cast(x=percentage / 100.0, dtype="float32")
        flag = percentage_tf > (exceeded / total)
        return flag

    @staticmethod
    def _pad_waveform_tf(
        waveform,
        sample_rate,
        stft_window_seconds,
        stft_hop_seconds,
        patch_window_seconds,
        patch_hop_seconds,
    ):
        """Pads waveform with zeros to fit required FFT length.

        Args:
            waveform: The normalised sound file.
            sample_rate: Desired sample rate in Hz.
            stft_window_seconds: Feature window length.
            stft_hop_seconds: The increment after each feature.
            patch_window_seconds: The window length of a sample.
            patch_hop_seconds: The increment after each sample.

        Returns:
            Padded waveform.
        """
        samples = tf.shape(waveform)[0]
        min_waveform_time = (
            patch_window_seconds + stft_window_seconds - stft_hop_seconds
        )
        min_samples = tf.cast(x=min_waveform_time * sample_rate, dtype="int32")
        # Handle waveform being less than minimum length
        if samples < min_samples:
            padding = tf.cast(x=min_samples - samples, dtype="float32")
            num_samples_after_first_patch = tf.cast(x=0.0, dtype="float32")
        else:
            padding = tf.cast(x=0.0, dtype="float32")
            num_samples_after_first_patch = tf.cast(
                x=samples - min_samples, dtype="float32"
            )
        # Pad
        hop_samples = tf.cast(
            x=patch_hop_seconds * sample_rate, dtype="float32"
        )
        num_hops_after_first_patch = tf.cast(
            x=tf.math.ceil(num_samples_after_first_patch / hop_samples),
            dtype="float32",
        )
        padding = (
            padding
            + (hop_samples * num_hops_after_first_patch)
            - num_samples_after_first_patch
        )
        padding = tf.cast(x=padding, dtype="int32")
        # Pad
        padded_waveform = tf.pad(
            tensor=waveform,
            paddings=[[0, padding]],
            mode="CONSTANT",
            constant_values=0.0,
        )
        return padded_waveform

    @staticmethod
    def _waveform_windows_tf(
        waveform, sample_rate, patch_window_seconds_padded, patch_hop_seconds
    ):
        """Divide a waveform into windows.
        # TODO
        Does not perfectly divide in the same way as YAMNet.
        Seems to works if input waveform is 10 seconds.

        Args:
            waveform: Waveform normalised to -1, 1.
            sample_rate: Sample rate in Hz.
            patch_window_seconds_padded: Patch window with padding.
            patch_hop_seconds: Patch step.

        Returns:
            Waveform divided into windows.
        """
        waveform_window = int(round(sample_rate * patch_window_seconds_padded))
        waveform_hop = int(round(sample_rate * patch_hop_seconds))
        waveform_windows = tf.signal.frame(
            signal=waveform,
            frame_length=waveform_window,
            frame_step=waveform_hop,
            axis=0,
        )
        return waveform_windows

    @staticmethod
    def load_sound_tf(
        filename,
        desired_channels,
        sample_rate,
        data_type,
        patch_window_seconds,
        std_mean,
        std_sd,
    ):
        """Load sound.

        Args:
            filename: The sound file.
            desired_channels: Get input from these channels.
            sample_rate: Desired sample rate in Hz.
            data_type: Date type of sound file as string.
            patch_window_seconds: The window length of a sample.
            std_mean: Standardisation mean. If None normalise input.
            std_sd: Standardisation standard deviation. If None normalise input.

        Returns:
            Arrays consisting representing sound.
        """
        waveform, _ = sf.read(file=filename, always_2d=True, dtype="float32")
        waveform = waveform[:, desired_channels]
        waveform = tf.convert_to_tensor(value=waveform, dtype="float32")
        if waveform.shape[-1] == 0:
            # Deal with zero length audio
            # {0} does not print during training
            print(f"Replaced zero length audio file {filename}.")
            filler_shape = (int(sample_rate * patch_window_seconds),)
            waveform = tf.zeros(shape=filler_shape, dtype=data_type)
        if std_mean and std_sd:
            waveform = (waveform - std_mean) / std_sd
        return waveform

    @staticmethod
    def _onehot_tf(headers):
        """Create onehot encoding given header of CSV filelist.

        Args:
            headers: A header from CSV. First index is filename.
            Subsequent header values are class names.

        Returns:
            Dictionary in following format {"class01": [1., 0.], ...}.
        """
        class_dict = dict()
        for index, class_name in enumerate(headers[1:]):
            class_vector = [0.0] * len(headers[1:])
            class_vector[index] = 1.0
            class_vector = tf.convert_to_tensor(
                value=class_vector, dtype="float32"
            )
            class_dict[class_name] = class_vector
        return class_dict

    @staticmethod
    def _class_weights_tf(class_count):
        """Get class weights based on count.

        Args:
            class_count: Tensor counting instances of every class.

        Returns:
            Dictionary containing class weights.
        """
        total = tf.math.reduce_sum(input_tensor=class_count)
        class_weights_values = (1 / class_count) * (total / 2.0)
        class_weights = dict()
        for index, value in enumerate(class_weights_values):
            class_weights[index] = float(value)
        return class_weights

    @staticmethod
    def _get_dataset_tf(filelist):
        """Get a list of files, labels and count number of classes.

        Args:
            filelist: A CSV containing the list of files and labels.

        Returns:
            A list of tuples in the following format.
            [(<sound file>, <onehot>)]
            E.g. [("class_a/sound_a_01.wav", [1., 0.]), ...]
            Dictionary mapping class names to onehot encoding.
            Tensor counting instances of every class.
        """
        with open(file=filelist, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            onehot_dict = None
            class_count = None
            dataset_list = list()
            for index, row in enumerate(reader):
                if index == 0:
                    # Convert first row into onehot encoding
                    onehot_dict = DataReaderTune._onehot_tf(headers=row)
                    # Initialise variable to count number of classes
                    sample_class = list(onehot_dict.values())
                    class_count = tf.zeros(
                        shape=sample_class[0].shape, dtype="float32"
                    )
                else:
                    filename = row[0]
                    vector = [int(x) for x in row[1:]]
                    vector = tf.convert_to_tensor(value=vector, dtype="float32")
                    dataset_list.append((filename, vector))
                    class_count += vector
            return dataset_list, onehot_dict, class_count
