"""
Calculate AST embeddings.
"""

import argparse
import glob
import librosa
import numpy as np
import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor


# AST hyperparameters
# device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_map = "auto"
model_dtype = "auto"


def _prepare_ast_model(model, activations, cls_vector):
    """Load the AST model and modify it so that embeddings
    can be extracted.

    Args:
        model: Model which will be modified.
        activations: Save embedding to this dictionary.
        cls_vector: Save embedding to a dictionary using this key.

    Returns:
        The model
    """

    # This function is to extract values from specified layer
    def get_activation(name):
        def hook(model_param, input_param, output):
            activations[name] = output.detach()

        return hook

    # Hook to connect dict() and layer to be extracted
    model.classifier.layernorm.register_forward_hook(
        hook=get_activation(cls_vector)
    )
    return model


def _main(filedir, ast_model, ast_npy):
    """Calculate AST embeddings.

    Args:
        filedir: Regex file path of audio. E.g. "*.wav".
        ast_model: AST model filepath.
        ast_npy: Path for saving AST embeddings.
    """
    filelist = glob.glob(pathname=filedir)
    filelist.sort()
    processor = ASTFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path=ast_model
    )
    model = ASTForAudioClassification.from_pretrained(
        pretrained_model_name_or_path=ast_model,
        dtype=model_dtype,
        device_map=device_map,
    )
    # activations and cls_vector are used to store embeddings
    activations = dict()
    cls_vector = "cls_vector"
    model = _prepare_ast_model(
        model=model,
        activations=activations,
        cls_vector=cls_vector,
    )
    model.eval()
    torch.inference_mode(mode=True)
    total = len(filelist)
    list_of_embeddings = list()
    for index, filename in enumerate(filelist):
        y, _ = librosa.load(
            path=filename, sr=processor.sampling_rate, mono=True
        )
        inputs = processor(
            raw_speech=y,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        )["input_values"].to(device=model.device)
        # The embeddings are extracted after a forward pass
        _ = model(input_values=inputs)
        embeddings = activations[cls_vector].detach().cpu().numpy()
        list_of_embeddings.append(embeddings)
        print(filename, index + 1, "of", total)
    concat_array = np.concatenate(list_of_embeddings, axis=0)
    np.save(file=ast_npy, arr=concat_array, allow_pickle=True)


def _command_line():
    """Process command line arguments into dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate AST embeddings.")
    parser.add_argument(
        "--filedir",
        metavar="S",
        type=str,
        required=True,
        dest="filedir",
        help='Regex file path of audio. E.g. "*.wav".',
    )
    parser.add_argument(
        "--ast_model",
        metavar="S",
        type=str,
        required=True,
        dest="ast_model",
        help="AST model filepath.",
    )
    parser.add_argument(
        "--ast_npy",
        metavar="S",
        type=str,
        required=True,
        dest="ast_npy",
        help="Path for saving AST embeddings.",
    )
    collected_arguments = vars(parser.parse_args())
    return collected_arguments


if __name__ == "__main__":
    arguments = _command_line()
    _main(**arguments)
