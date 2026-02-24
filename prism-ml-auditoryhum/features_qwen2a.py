"""
Calculate QWEN2-Audio embeddings.
"""

import argparse
import glob
import librosa
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2AudioForConditionalGeneration,
)


# QWEN2-Audio hyperparameters
# device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_map = "auto"
model_dtype = "auto"
# model_dtype = torch.bfloat16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# quant_config = BitsAndBytesConfig()


def _main(filedir, qwen2a_model, qwen2a_npy):
    """Calculate QWEN2-Audio embeddings.

    Args:
        filedir: Regex file path of audio. E.g. "*.wav".
        qwen2a_model: QWEN2-Audio model filepath.
        qwen2a_npy: Path for saving QWEN2-Audio embeddings.
    """
    filelist = glob.glob(pathname=filedir)
    filelist.sort()
    total = len(filelist)
    list_of_embeddings = list()
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=qwen2a_model
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=qwen2a_model,
        quantization_config=quant_config,
        dtype=model_dtype,
        device_map=device_map,
    )
    model.eval()
    torch.inference_mode(mode=True)
    for index, filename in enumerate(filelist):
        y, _ = librosa.load(
            path=filename,
            sr=processor.feature_extractor.sampling_rate,
            mono=True,
        )
        inputs = processor.feature_extractor(
            raw_speech=y,
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
        ).to(device=model.device)
        audio_outputs = model.audio_tower(
            input_features=inputs["input_features"]
        )
        # Get the audio embeddings
        audio_embeddings = audio_outputs.last_hidden_state
        # Get the audio tokens for LLM
        audio_tokens_for_llm = model.multi_modal_projector(
            audio_features=audio_embeddings
        )
        # Sum audio embeddings
        sum_audio_embeddings = torch.sum(input=audio_embeddings, dim=1)
        sum_audio_embeddings = (
            sum_audio_embeddings.detach().cpu().float().numpy()
        )
        list_of_embeddings.append(sum_audio_embeddings)
        print(filename, index + 1, "of", total)
    concat_array = np.concatenate(list_of_embeddings, axis=0)
    np.save(file=qwen2a_npy, arr=concat_array, allow_pickle=True)


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate QWEN2-Audio embeddings."
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
        "--qwen2a_model",
        metavar="S",
        type=str,
        required=True,
        dest="qwen2a_model",
        help="QWEN2-Audio model filepath.",
    )
    parser.add_argument(
        "--qwen2a_npy",
        metavar="S",
        type=str,
        required=True,
        dest="qwen2a_npy",
        help="Path for saving QWEN2-Audio embeddings.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
