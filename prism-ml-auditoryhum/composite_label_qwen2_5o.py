"""
Prompt QWEN2.5-Omni.
"""

import argparse
import copy
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
conversation_text = "Prompt text."
# The following text in system_role is required for Qwen to work properly
system_role = (
    "You are a neutral, objective informational engine. Provide factual, "
    "completely unbiased responses. Do not use conversational filler, "
    "greetings, pleasantries, or concluding remarks. Answer the user's "
    "prompt directly, using clear and structured language. If a topic "
    "is controversial or lacks consensus, present the main viewpoints "
    "neutrally without taking a side."
)
conversation = [
    {"role": "system", "content": [{"type": "text", "text": system_role}]},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": conversation_text,
            },
        ],
    },
]
model_dtype = "auto"
# model_dtype = torch.bfloat16
"""
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
"""
# quant_config = BitsAndBytesConfig()
model_seed = 100


def prompt_helper(
    processor,
    model,
    conversation,
    prompt,
    max_new_tokens,
):
    """Prompt helper.

    Args:
        processor: Processor to preprocess input.
        model: The model.
        conversation: The prompt datastructure.
        prompt: Prompt text.
        max_new_tokens: Max output tokens.
    """
    # Create a deep copy of the conversation
    conv_copy = copy.deepcopy(conversation)
    conv_copy[1]["content"][0]["text"] = prompt
    # Preprocess text
    text = processor.apply_chat_template(
        conversations=conv_copy,
        add_generation_prompt=True,
        tokenize=False,
    )
    # Convert prompt into correct format for model
    inputs = processor(
        text=[text], return_tensors="pt", padding="max_length"
    ).to(device=model.device)
    # Get text
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )
    # Translate ids to text
    generate_ids2 = generate_ids[:, inputs.input_ids.size(1) :]
    response = processor.batch_decode(
        sequences=generate_ids2,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(response)


def _main(qwen2_5o_model, prompt):
    """Prompt QWEN2.5-Omni.

    Args:
        qwen2_5o_model: QWEN2.5-Omni model filepath.
        prompt: Prompt text.
    """
    label_qwen2a.set_seed(seed=model_seed)
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=qwen2_5o_model
    )
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=qwen2_5o_model,
        # quantization_config=quant_config,
        dtype=model_dtype,
        device_map=device_map,
    )
    model.eval()
    torch.inference_mode(mode=True)
    prompt_helper(
        processor=processor,
        model=model,
        conversation=conversation,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="Prompt QWEN2.5-Omni.")
    parser.add_argument(
        "--qwen2_5o_model",
        metavar="S",
        type=str,
        required=True,
        dest="qwen2_5o_model",
        help="QWEN2.5-Omni model filepath.",
    )
    parser.add_argument(
        "--prompt",
        metavar="S",
        type=str,
        required=False,
        dest="prompt",
        help="Text prompt.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
