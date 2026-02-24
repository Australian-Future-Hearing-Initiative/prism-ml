"""
Calculate QWEN2.5-Omni labels.
"""

import argparse
import copy
import glob
import librosa
import numpy as np
import string
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
)


import label_qwen2a


# QWEN2.5-Omni hyperparameters
# device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_map = "auto"
max_new_tokens = 64
# Spped up calculations
use_cache = True
# Disables randomness
do_sample = False
# Disables randomness
num_beams = 1
# Creativity parameter
temperature = 1.0
conversation_text = "Describe the auditory scene using word pairs. Separate \
each pair with a comma."
# The following text in system_role is required for Qwen to work properly
system_role = "You are Qwen, a virtual human developed by the Qwen \
Team, Alibaba Group, capable of perceiving auditory and visual inputs, \
as well as generating text and speech."
conversation = [
    {"role": "system", "content": [{"type": "text", "text": system_role}]},
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                # Required placeholder
                "audio": "placeholder",
            },
            {
                "type": "text",
                "text": conversation_text,
            },
        ],
    },
]
model_dtype = "auto"
# model_dtype = torch.bfloat16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# quant_config = BitsAndBytesConfig()
model_seed = 100


def label_helper_v2(
    processor,
    model,
    filelist,
    conversation,
    max_new_tokens,
    use_cache,
    do_sample,
    num_beams,
    temperature,
    annot_list,
):
    """Loop through data and get labels.

    Args:
        processor: Processor to preprocess input.
        model: The model.
        filelist: List of files to iterate.
        conversation: The prompt list.
        max_new_tokens: Max output tokens.
        use_cache: Speed up.
        do_sample: Controls deterministic output.
        num_beams: Controls deterministic output.
        temperature: Creativity parameter.
        annot_list: List of provided annotations.

    Returns:
        All labels as a list.
    """
    label_concat = list()
    total = len(filelist)
    for index, filename in enumerate(filelist):
        y, _ = librosa.load(
            path=filename,
            sr=processor.feature_extractor.sampling_rate,
            mono=True,
        )
        # Create a deep copy of the conversation just for this iteration
        conv_copy = copy.deepcopy(conversation)
        conv_copy[1]["content"][0]["audio"] = filename
        # Apply any human annotations
        if annot_list:
            if annot_list[index].strip():
                conv_copy[1]["content"][1]["text"] = (
                    conv_copy[1]["content"][1]["text"]
                    + " "
                    + annot_list[index].strip()
                )
        text = processor.apply_chat_template(
            conversations=conv_copy,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(
            text=[text], audio=[y], return_tensors="pt", padding="max_length"
        ).to(device=model.device)
        # Get labels
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            # Disables randomness
            do_sample=do_sample,
            # Disables randomness
            num_beams=num_beams,
            temperature=temperature,
        )
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
        response = processor.batch_decode(
            sequences=generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        # Cache label
        response = label_qwen2a.strip_for_saving(text=response)
        label_concat.append(response)
        print(filename, index + 1, "of", total, "response:", response)
    return label_concat


def _main(filedir, qwen2_5o_model, annotations, text_label_csv):
    """Calculate QWEN2.5-Omni labels.

    Args:
        filedir: Regex file path of audio. E.g. "*.wav".
        qwen2_5o_model: QWEN2.5-Omni model filepath.
        annotations: The path of the annotations CSV.
        text_label_csv: The text labels are saved to this CSV.
    """
    label_qwen2a.set_seed(seed=model_seed)
    filelist = glob.glob(pathname=filedir)
    filelist.sort()
    annot_list = label_qwen2a.read_annot_data(annotations=annotations)
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=qwen2_5o_model
    )
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=qwen2_5o_model,
        quantization_config=quant_config,
        dtype=model_dtype,
        device_map=device_map,
    )
    model.eval()
    torch.inference_mode(mode=True)
    label_concat = label_helper_v2(
        processor=processor,
        model=model,
        conversation=conversation,
        filelist=filelist,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        do_sample=do_sample,
        num_beams=num_beams,
        temperature=temperature,
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
        description="Calculate QWEN2.5-Omni labels."
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
        "--qwen2_5o_model",
        metavar="S",
        type=str,
        required=True,
        dest="qwen2_5o_model",
        help="QWEN2.5-Omni model filepath.",
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
        help="The text labels are saved this to CSV.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
