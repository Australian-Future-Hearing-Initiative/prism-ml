"""
Get text label CLAP embeddings.
"""

import argparse
import glob
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ClapModel, ClapProcessor


# CLAP hyperparameters
device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _text_to_set(text_label_csv):
    """Get list of lists text labels.

    Args:
        text_label_csv: The text labels are in this CSV.

    Returns:
        List of lists text labels.
    """
    lines = list()
    with open(file=text_label_csv, mode="r", encoding="utf-8") as label_file:
        index = 0
        for line in label_file:
            cleaned_line = line.split(",")
            # Drop any non ASCII text
            cleaned_line = [
                x.encode("ascii", "ignore").decode() for x in cleaned_line
            ]
            # Strip blanks from ends
            cleaned_line = [x.strip() for x in cleaned_line]
            # Force the text to be at most word pairs
            cleaned_line = [" ".join(x.split()[:2]) for x in cleaned_line]
            lines.append(cleaned_line)
            index += 1
            print(f"Labels {index}")
    return lines


def _text_label_embeddings(model, processor, text_label):
    """Calculate text label embeddings.

    Args:
        model: Clap model.
        processor: Clap data processor.
        text_label: The text labels as a list of lists.

    Returnd:
        Text label embeddings as a list of lists.
    """
    embeddings_list = list()
    for index, single_line in enumerate(text_label):
        inputs = processor(
            text=single_line, padding=True, truncation=True, return_tensors="pt"
        ).to(device=model.device)
        outputs = model.get_text_features(**inputs)
        embeddings = outputs.detach().cpu().float().numpy()
        embeddings_list.append(embeddings)
        print(f"Embeddings {index + 1} of {len(text_label)}")
    return embeddings_list


def _compare_wav_text(audio_embeddings, text_embeddings):
    """Find highest cosine similarity score between audio and text embeddings.

    Args:
        audio_embeddings: List of audio embeddings.
        text_embeddings: List of text embeddings.

    Returnd:
        List of highest cosine scores.
        List of label indices for highest score.
        Text embeddings array for top scores.
    """
    top_score_list = list()
    top_index_list = list()
    top_text_embeddings_list = list()
    length = audio_embeddings.shape[0]
    for index in range(length):
        cosine_scores = cosine_similarity(
            X=audio_embeddings[index : index + 1], Y=text_embeddings[index]
        )
        top_index = int(cosine_scores.argmax())
        top_score_list.append(float(cosine_scores[0, top_index]))
        top_index_list.append(top_index)
        top_text_embeddings_list.append(text_embeddings[index][top_index])
        print(f"Cosine scores {index + 1} of {length}")
    top_text_embeddings_list = np.stack(arrays=top_text_embeddings_list, axis=0)
    return top_score_list, top_index_list, top_text_embeddings_list


def _main(filedir, clap_model, clap_npy, text_label_csv, top_label_scores_csv):
    """Get text label CLAP embeddings.

    Args:
        filedir: Regex file path of audio. E.g. "*.wav".
        clap_model: CLAP model filepath.
        clap_npy: Path for existing CLAP embeddings.
        text_label_csv: The text labels from MLLM.
        top_label_scores_csv: Path for saving scores.
        chosen_labels_csv: Path for saving chosen labels and scores.
    """
    filelist = glob.glob(pathname=filedir)
    filelist.sort()
    processor = ClapProcessor.from_pretrained(
        pretrained_model_name_or_path=clap_model
    )
    model = ClapModel.from_pretrained(
        pretrained_model_name_or_path=clap_model,
        device_map=device_map,
    )
    model.eval()
    torch.inference_mode(mode=True)
    audio_embeddings = np.load(file=clap_npy, allow_pickle=True)
    text_label = _text_to_set(text_label_csv=text_label_csv)
    text_embeddings = _text_label_embeddings(
        model=model,
        processor=processor,
        text_label=text_label,
    )
    top_score_list, top_index_list, top_text_embeddings = _compare_wav_text(
        audio_embeddings=audio_embeddings, text_embeddings=text_embeddings
    )
    top_text_labels = [
        text_label[index][x] for index, x in enumerate(top_index_list)
    ]
    print(f"Mean CLAP score: {np.mean(a=top_score_list)}")
    with open(
        file=top_label_scores_csv, mode="w", encoding="utf-8"
    ) as label_file:
        for index, filename in enumerate(filelist):
            line = f"{(index+ 1)}, {filelist[index]}, "
            line += f"{top_text_labels[index]}, {top_score_list[index]}"
            label_file.write(line + "\n")


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Get text label CLAP embeddings."
    )
    parser.add_argument(
        "--filedir",
        metavar="S",
        type=str,
        required=True,
        dest="filedir",
        help='Regex file path of audio. E.g. "*.wav".',
    )
    parser.add_argument(
        "--clap_model",
        metavar="S",
        type=str,
        required=True,
        dest="clap_model",
        help="CLAP model filepath.",
    )
    parser.add_argument(
        "--clap_npy",
        metavar="S",
        type=str,
        required=True,
        dest="clap_npy",
        help="Path for existing CLAP embeddings.",
    )
    parser.add_argument(
        "--text_label_csv",
        metavar="S",
        type=str,
        required=True,
        dest="text_label_csv",
        help="The text labels from MLLM.",
    )
    parser.add_argument(
        "--top_label_scores_csv",
        metavar="S",
        type=str,
        required=True,
        dest="top_label_scores_csv",
        help="Path for saving chosen labels and scores.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
