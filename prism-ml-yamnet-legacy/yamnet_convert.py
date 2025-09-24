"""
This module converts legacy YAMNet into SavedModel format.
"""

import params
import yamnet


if __name__ == "__main__":
    mymodel = yamnet.yamnet_frames_model(params=params.Params)
    # Get yamnet.h5 from https://storage.googleapis.com/audioset/yamnet.h5
    # Comment out the line below for random weights
    mymodel.load_weights(filepath="yamnet.h5")
    mymodel.export(filepath="yamnet_model")
