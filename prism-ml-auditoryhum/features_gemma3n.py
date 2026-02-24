"""
Calculate Gemma-3N embeddings.
"""

import argparse
import glob
import librosa
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3nForConditionalGeneration,
)


# Gemma-3N hyperparameters
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


def _main(filedir, gemma3n_model, gemma3n_npy):
    """Calculate Gemma-3N embeddings.

    Args:
        filedir: Regex file path of audio. E.g. "*.wav".
        gemma3n_model: Gemma-3N model filepath.
        gemma3n_npy: Path for saving Gemma-3N embeddings.
    """
    filelist = glob.glob(pathname=filedir)
    filelist.sort()
    total = len(filelist)
    list_of_embeddings = list()
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
    for index, filename in enumerate(filelist):
        y, _ = librosa.load(
            path=filename,
            sr=processor.feature_extractor.sampling_rate,
            mono=True,
        )
        inputs = processor(
            text="",
            audio=[y],
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
        ).to(device=model.device)
        # Use mask filled with False
        # A mask filled with True will cause no values to be computed
        audio_mel_mask = torch.full(
            size=inputs["input_features_mask"].shape,
            fill_value=False,
            dtype=inputs["input_features_mask"].dtype,
        ).to(device=model.device)
        audio_outputs = model.model.audio_tower(
            audio_mel=inputs["input_features"], audio_mel_mask=audio_mel_mask
        )
        # Get the audio embeddings
        audio_embeddings = audio_outputs[0]
        # Get the audio tokens for LLM
        audio_tokens_for_llm = model.model.embed_audio.embedding_projection(
            input=audio_embeddings
        )
        # Sum audio embeddings
        sum_audio_embeddings = torch.sum(input=audio_embeddings, dim=1)
        sum_audio_embeddings = (
            sum_audio_embeddings.detach().cpu().float().numpy()
        )
        list_of_embeddings.append(sum_audio_embeddings)
        print(filename, index + 1, "of", total)
    concat_array = np.concatenate(list_of_embeddings, axis=0)
    np.save(file=gemma3n_npy, arr=concat_array, allow_pickle=True)


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate Gemma-3N embeddings."
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
        "--gemma3n_model",
        metavar="S",
        type=str,
        required=True,
        dest="gemma3n_model",
        help="Gemma-3N model filepath.",
    )
    parser.add_argument(
        "--gemma3n_npy",
        metavar="S",
        type=str,
        required=True,
        dest="gemma3n_npy",
        help="Path for saving Gemma-3N embeddings.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
