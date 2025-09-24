"""
Test a dataset by producing metrics.
"""

import argparse
import itertools
import tensorflow as tf

import datareader_tune
import train_transfer


# Disable GPU on TensorFlow
# TF crashes with JIT error if CUDA CUDNN not properly set up
# Stick with CPU just to be safe
tf.config.set_visible_devices(devices=[], device_type="GPU")

# Do not augment or shuffle for test
shuffle = False
augmentations = None


def create_conf_matrix(class_num):
    """Create confusion matrix.
    Note: Does not support multi-label results.

    Args:
        class_num: Number of classes.

    Returns:
        List of lists to save confusion matrix
        in the following format: [[0,0,0],[0,0,0],[0,0,0]]
    """
    conf_matrix = list()
    for _ in range(class_num + 1):
        conf_matrix.append([0] * class_num)
    return conf_matrix


def add_conf_matrix_tf(truth, predicted, conf_matrix):
    """Add metrics to confusion matrix.
    Note: Does not support multi-label results.

    Args:
        truth: Ground truth.
        predicted: Predicted value.
        conf_matrix: Confusion matrix.
    """
    # Find index of cells with 1.0 from truth
    truth_index = tf.math.argmax(input=truth, axis=1)
    # Find positions where no label provided and replace
    truth_replace = tf.reduce_sum(input_tensor=truth, axis=1) == 0.0
    truth_index_new = tf.where(condition=truth_replace, x=-1, y=truth_index)
    # Find index of highest confidence value from predicted
    predicted_index = tf.math.argmax(input=predicted, axis=1)
    combined_indices = zip(truth_index_new, predicted_index)
    # Add the classification to confusion matrix
    for tindex, pindex in combined_indices:
        conf_matrix[tindex][pindex] += 1


def print_conf_matrix_tf(class_names, conf_matrix):
    """Print confusion matrix and accuracy.
    Note: Does not support multi-label results.

    Args:
        class_names: List of class names.
        conf_matrix: Confusion matrix.
    """
    first_row = ", ".join(["class"] + class_names)
    print(first_row)
    for index, single_name in enumerate(class_names + ["<Undefined>"]):
        row_numbers_string = [str(x) for x in conf_matrix[index]]
        more_rows = ", ".join([single_name] + row_numbers_string)
        print(more_rows)
    correct = 0
    total = 0
    for index in range(len(conf_matrix) - 1):
        # The range is -1 to account for <undefined values>
        correct += conf_matrix[index][index]
        total += sum(conf_matrix[index])
    if total > 0:
        print("Accuracy", correct / total)
    else:
        print("Accuracy", correct, "div", total)
    print()


def add_pr_dict_tf(truth, predicted, threshold, pr_dict):
    """Add TP, TN, FP, FN to precision/recall dictionary.

    Args:
        truth: Ground truth.
        predicted: Predicted value.
        threshold: The threshold for detection.
        pr_dict: Precision/recall dictionary.
    """
    # Find TP, FP, FN, TN
    # Do this by multiplying pred by 2 and summing with truth
    # 3=TP, 2=FP, 1=FN, 0=TN
    predicted2 = predicted * 2.0
    compared = predicted2 + truth
    for index in range(truth.shape[1]):
        tp_tensor = tf.cast(x=compared[:, index] == 3, dtype="int32")
        fp_tensor = tf.cast(x=compared[:, index] == 2, dtype="int32")
        fn_tensor = tf.cast(x=compared[:, index] == 1, dtype="int32")
        tn_tensor = tf.cast(x=compared[:, index] == 0, dtype="int32")
        tp = tf.math.reduce_sum(input_tensor=tp_tensor)
        fp = tf.math.reduce_sum(input_tensor=fp_tensor)
        fn = tf.math.reduce_sum(input_tensor=fn_tensor)
        tn = tf.math.reduce_sum(input_tensor=tn_tensor)
        pr_dict[(threshold, index)]["tp"] += int(tp)
        pr_dict[(threshold, index)]["fp"] += int(fp)
        pr_dict[(threshold, index)]["fn"] += int(fn)
        pr_dict[(threshold, index)]["tn"] += int(tn)


def create_pr_dict(threshold_list, class_num):
    """Create dictionary for data used to compute precision/recall.

    Args:
        threshold_list: List of threshold values.
        class_num: Number of classes.

    Returns:
        Dictionary to save precision and recall
        in the following format: dict[(thresh, index)]
        = {"TP":0,"TN":0,"FP":0,"FN":0}
    """
    pr_dict = dict()
    thresh_classes = itertools.product(threshold_list, range(class_num))
    for index in thresh_classes:
        pr_dict[index] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    return pr_dict


def print_pr_tf(threshold_list, class_names, pr_dict):
    """Print precision/recall.

    Args:
        threshold_list: List of threshold values.
        class_names: List of class names.
        pr_dict: Precision/recall dictionary.
    """
    eps = tf.keras.backend.epsilon()
    # Store average precision
    ap_list = list()
    for cindex, cname in enumerate(class_names):
        print(cname)
        # Store precision
        precision_list = list()
        # For each class print threshold and metrics
        print("Threshold, Precision, Recall, TP, TN, FP, FN")
        for thresh_val in threshold_list:
            class_metrics = pr_dict[(thresh_val, cindex)]
            tp = class_metrics["tp"]
            tn = class_metrics["tn"]
            fp = class_metrics["fp"]
            fn = class_metrics["fn"]
            precision = (tp + eps) / (tp + fp + eps)
            recall = (tp + eps) / (tp + fn + eps)
            print(
                f"{thresh_val:.5f}, {precision:.5f}, "+
                f"{recall:.5f}, {tp}, {tn}, {fp}, {fn}"
            )
            precision_list.append(precision)
        # Print and store ap value
        ap_value = sum(precision_list) / len(precision_list)
        ap_list.append(ap_value)
        print(f"{cname} average precision, {ap_value} \n")
    mean_ap = sum(ap_list) / len(ap_list)
    print("Mean average precision:", mean_ap)


def _main_tf(model_file, filelist, threshold):
    """Test a dataset by producing metrics.

    Args:
        model_file: Model filepath.
        filelist: A CSV containing the list of files and labels.
        threshold: Comma separated confidence thresholds.
        Typical range [0-1].
    """
    # Prepare test data
    test_data = datareader_tune.DataReaderTune(
        filelist=filelist,
        desired_channels=train_transfer.desired_channels,
        sample_rate=train_transfer.sample_rate,
        data_type=train_transfer.data_type,
        stft_window_seconds=train_transfer.stft_window_seconds,
        stft_hop_seconds=train_transfer.stft_hop_seconds,
        patch_window_seconds=train_transfer.patch_window_seconds,
        patch_window_seconds_padded=train_transfer.patch_window_seconds_padded,
        patch_hop_seconds=train_transfer.patch_hop_seconds,
        mel_bands=train_transfer.mel_bands,
        mel_min_hz=train_transfer.mel_min_hz,
        mel_max_hz=train_transfer.mel_max_hz,
        log_offset=train_transfer.log_offset,
        sil_threshold=train_transfer.sil_threshold,
        sil_percentage=train_transfer.sil_percentage,
        sil_vector=train_transfer.sil_vector,
        std_mean=train_transfer.std_mean,
        std_sd=train_transfer.std_sd,
        shuffle=shuffle,
        augmentations=augmentations,
    )
    # Thresholds
    threshold_list = [float(x) for x in threshold.split(",")]
    # Class names
    class_names = list(test_data.get_onehot().keys())
    # Create precision/recall dictionary
    pr_dict = create_pr_dict(
        threshold_list=threshold_list, class_num=len(class_names)
    )
    # Create confusion matrix
    conf_matrix = create_conf_matrix(class_num=len(class_names))
    # Load model
    model = tf.keras.models.load_model(filepath=model_file)
    for index, (waveform, label) in enumerate(test_data):
        predicted_raw = model(waveform)
        # Iterate through all thresholds
        for thresh_val in threshold_list:
            # Process predicted and truth labels
            predicted = tf.cast(x=predicted_raw > thresh_val, dtype="float32")
            # Save results of comparison
            add_pr_dict_tf(
                truth=label,
                predicted=predicted,
                threshold=thresh_val,
                pr_dict=pr_dict,
            )
        # Populate confusion matrix
        add_conf_matrix_tf(
            truth=label, predicted=predicted_raw, conf_matrix=conf_matrix
        )
        print("Sample", index + 1, "of", len(test_data))
    print_conf_matrix_tf(class_names=class_names, conf_matrix=conf_matrix)
    print_pr_tf(
        threshold_list=threshold_list, class_names=class_names, pr_dict=pr_dict
    )


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Test a dataset by producing \
                        metrics."
    )
    parser.add_argument(
        "--model_file",
        metavar="S",
        type=str,
        required=True,
        dest="model_file",
        help="Model filepath.",
    )
    parser.add_argument(
        "--filelist",
        metavar="S",
        type=str,
        required=True,
        dest="filelist",
        help="A CSV containing the list of files and labels.",
    )
    parser.add_argument(
        "--threshold",
        metavar="S",
        type=str,
        required=True,
        dest="threshold",
        help="Comma separated confidence thresholds. Typical range [0-1].",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main_tf(**arguments)
