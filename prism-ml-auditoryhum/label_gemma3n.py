"""
Calculate Gemma-3N labels.
"""

import argparse
import glob
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3nForConditionalGeneration,
)


import label_qwen2a


# Gemma-3N hyperparameters
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
system_role = "Describing an auditory scene."
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


def _main(filedir, gemma3n_model, annotations, text_label_csv):
    """Calculate Gemma-3N labels.

    Args:
        filedir: Regex file path of audio. E.g. "*.wav".
        gemma3n_model: Gemma-3N model filepath.
        annotations: The path of the annotations CSV.
        text_label_csv: The text labels are saved to this CSV.
    """
    label_qwen2a.set_seed(seed=model_seed)
    filelist = glob.glob(pathname=filedir)
    filelist.sort()
    annot_list = label_qwen2a.read_annot_data(annotations=annotations)
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=gemma3n_model
    )
    model = Gemma3nForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=gemma3n_model,
        quantization_config=quant_config,
        dtype=model_dtype,
        device_map=device_map,
    )
    model.eval()
    torch.inference_mode(mode=True)
    label_concat = label_qwen2a.label_helper(
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
    with open(file=text_label_csv, mode="w", encoding="utf-8") as label_file:
        label_file.write(label_concat)


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate Gemma-3N labels.")
    parser.add_argument(
        "--filedir",
        metavar="S",
        type=str,
        required=True,
        dest="filedir",
        help='Regex file path of audio. E.g. "*.wav".',
    )
    parser.add_argument(
        "--gemma3n_model",
        metavar="S",
        type=str,
        required=True,
        dest="gemma3n_model",
        help="Gemma-3N model filepath.",
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
