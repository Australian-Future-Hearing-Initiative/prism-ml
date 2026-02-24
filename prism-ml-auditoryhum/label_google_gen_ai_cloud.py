"""
Calculate Google Gen AI labels.
"""

import argparse
import glob
from google import genai
from google.genai import errors, types
import sys


import label_qwen2a


# Google Cloud Gen AI parameters
conversation_text = "Describe the auditory scene using word pairs. Separate \
each pair with a comma."
max_output_tokens = 1024
model_seed = 100
temperature = 0.0
top_p = 1.0
config = types.GenerateContentConfig(
    max_output_tokens=max_output_tokens,
    temperature=temperature,
    seed=model_seed,
    top_p=top_p,
)


def label_online(
    filelist,
    genai_id,
    api_key,
    config,
    conversation_text,
    annot_list,
):
    """Loop through data and get labels online.

    Args:
        filelist: List of files to iterate.
        genai_id: The AI ID to pass to the API.
        api_key: API Key.
        config: Model config.
        conversation_text: The prompt query.
        annot_list: List of provided annotations.

    Returns:
        All labels as a list.
    """
    client = genai.Client(api_key=api_key)
    label_concat = list()
    total = len(filelist)
    for index, filename in enumerate(filelist):
        # Apply any annotations
        conversation_text_full = conversation_text
        if annot_list:
            if annot_list[index].strip():
                conversation_text_full = (
                    conversation_text_full + " " + annot_list[index].strip()
                )
        # This loop repeats requests which fail until complete
        not_complete = True
        while not_complete:
            try:
                # Send audio file
                audio_data = client.files.upload(file=filename)
                response = client.models.generate_content(
                    model=genai_id,
                    config=config,
                    contents=[conversation_text_full, audio_data],
                )
                not_complete = False
            except errors.ClientError as e:
                print(f"Google AI Studio client error: {e}")
                print(filename, index + 1, "of", total)
                # sys.exit(1)
            except Exception as e:
                print(f"Non client error: {e}")
                print(filename, index + 1, "of", total)
                # sys.exit(1)
        # Cache label
        response_stripped = label_qwen2a.strip_for_saving(text=response.text)
        label_concat.append(response_stripped)
        print(filename, index + 1, "of", total, "response:", response_stripped)
    return label_concat


def _main(filedir, genai_id, api_key, annotations, text_label_csv):
    """Calculate Google Cloud Gen AI labels.

    Args:
        filedir: Regex file path of audio. E.g. "*.wav".
        genai_id: The AI ID to pass to the API.
        api_key: API Key.
        annotations: The path of the annotations CSV.
        text_label_csv: The text labels are saved to this CSV.
    """
    label_qwen2a.set_seed(seed=model_seed)
    filelist = glob.glob(pathname=filedir)
    filelist.sort()
    annot_list = label_qwen2a.read_annot_data(annotations=annotations)
    label_concat = label_online(
        filelist=filelist,
        genai_id=genai_id,
        api_key=api_key,
        config=config,
        conversation_text=conversation_text,
        annot_list=annot_list,
    )
    label_qwen2a.write_label(
        text_label_csv=text_label_csv, label_concat=label_concat
    )


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate Google Cloud Gen AI labels."
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
        "--genai_id",
        metavar="S",
        type=str,
        required=True,
        dest="genai_id",
        help="The AI ID to pass to the API.",
    )
    parser.add_argument(
        "--api_key",
        metavar="S",
        type=str,
        required=True,
        dest="api_key",
        help="API key.",
    )
    parser.add_argument(
        "--annotations",
        metavar="S",
        type=str,
        required=False,
        dest="annotations",
        help="The path of the annotations CSV.",
    )
    parser.add_argument(
        "--text_label_csv",
        metavar="S",
        type=str,
        required=True,
        dest="text_label_csv",
        help="The text labels are saved to this CSV.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
