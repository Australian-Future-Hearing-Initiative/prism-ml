# Introduction

Load the original YAMNet model definitions and re-save into SavedModel format.
Starting with TensorFlow 2.16.1, the included version of Keras is V3.
There are compatibility problems with loading YAMNet model definitions using Keras V3.
This version uses TF-Keras which is Keras V2.
SavedModel format allows the model to be used in Keras V3.

This codebase was used in the AHEAD-DS OpenYAMNet/YAMNet+ paper.

* [Website](https://github.com/Australian-Future-Hearing-Initiative)
* [Paper](https://arxiv.org/abs/2508.10360)
* [Code](https://github.com/Australian-Future-Hearing-Initiative/prism-ml/prism-ml-yamnetp-tune)
* [Dataset AHEAD-DS](https://huggingface.co/datasets/hzhongresearch/ahead_ds)
* [Dataset AHEAD-DS unmixed](https://huggingface.co/datasets/hzhongresearch/ahead_ds_unmixed)
* [Models](https://huggingface.co/hzhongresearch/yamnetp_ahead_ds)

# Setup and run conversion

```
# Setup
sudo apt update
sudo apt install --yes git python3 python3-pip python3-venv wget
python3 -m venv env_tflegacy
source env_tflegacy/bin/activate
git clone https://github.com/Australian-Future-Hearing-Initiative/prism-ml.git
cd prism-ml/prism-ml-yamnet-legacy
pip install --upgrade pip
pip install --requirement requirements.txt

# Download weights from
wget -O yamnet.h5 https://storage.googleapis.com/audioset/yamnet.h5

# Load YAMNet and re-save
python3 yamnet_convert.py
```

# Licence

Licenced under Apache 2.0. See [LICENCE.txt](LICENCE.txt).

Attribution.

```
@misc{zhong2026datasetmodelauditoryscene,
      title={A dataset and model for auditory scene recognition for hearing devices: AHEAD-DS and OpenYAMNet}, 
      author={Henry Zhong and Jörg M. Buchholz and Julian Maclaren and Simon Carlile and Richard Lyon},
      year={2026},
      eprint={2508.10360},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2508.10360}, 
}
```
