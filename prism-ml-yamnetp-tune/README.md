# Introduction

Train OpenYAMNet/YAMNet+ using YAMNet.

* [Website](https://github.com/Australian-Future-Hearing-Initiative)
* [Paper](https://arxiv.org/abs/2508.10360)
* [Code](https://github.com/Australian-Future-Hearing-Initiative/prism-ml/prism-ml-yamnetp-tune)
* [Dataset AHEAD-DS](https://huggingface.co/datasets/hzhongresearch/ahead_ds)
* [Dataset AHEAD-DS unmixed](https://huggingface.co/datasets/hzhongresearch/ahead_ds_unmixed)
* [Models](https://huggingface.co/hzhongresearch/yamnetp_ahead_ds)

# Setup, tune and perform transfer learning

```
# Setup
sudo apt update
sudo apt install --yes git git-lfs gzip python3 python3-pip python3-venv tar wget
python3 -m venv env_yamnet
source env_yamnet/bin/activate
git clone https://github.com/Australian-Future-Hearing-Initiative/prism-ml.git
cd prism-ml/prism-ml-yamnetp-tune
pip install --upgrade pip
pip install --requirement requirements.txt

# Download YAMNet
wget -O yamnet-tensorflow2-yamnet-v1.tar.gz https://www.kaggle.com/api/v1/models/google/yamnet/tensorFlow2/yamnet/1/download
mkdir yamnet_model
tar -xvzf yamnet-tensorflow2-yamnet-v1.tar.gz --directory yamnet_model

# Download AHEAD-DS training, validation and testing data into working directory
git clone https://huggingface.co/datasets/hzhongresearch/ahead_ds
mv ahead_ds/*.wav .
mv ahead_ds/*.csv .
rm -rf ahead_ds

# Tune using transfer learning example
python3 train_transfer.py --log_directory=log --train_filelist=ahead_ds_training.csv --val_filelist=ahead_ds_validation.csv --existing_model_file=yamnet_model --new_model_file=yamnetp_ahead_ds.keras --epochs=100

# Inference example
python3 inference.py --model_file=yamnetp_ahead_ds.keras --sound_file=cocktail_party_00001.wav

# Test example
python3 test.py --model_file=yamnetp_ahead_ds.keras --filelist=ahead_ds_testing.csv --threshold="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

# Convert to LiteRT
python3 convert_to_litert.py --keras_file=yamnetp_ahead_ds.keras --liteRT_file=yamnetp_ahead_ds.tflite
```

# Data format
The data needs to be in WAV format. Files are expected to be in 16 bit, 16 kHz mono format. Each dataset requires a CSV in the following format.

```
File,class_a,class_b
class_a_01.wav,1,0
class_a_02.wav,1,0
class_b_01.wav,0,1
class_b_02.wav,0,1
```

# Monitoring using Tensorboard

```
# Tensorboard records logs of the training process
tensorboard --logdir="./log" --port 6006
# Open http://localhost:6006/ in browser after starting Tensorboard
```

# Notes on maintaining code
The original YAMNet was written when TensorFlow V2 and Keras V2 code could be used interchangeably. The code for this version of YAMNet was written when Keras V3 was released and backwards compatibility was lost with TensorFlow V2 and Keras V2. All functions which use TensorFlow operators, casts and initialisations are postpended with \_tf. They work for now but need to be refactored out and replaced with equivalent Keras V3 operators, casts and initialisations to ensure compatibility in the future.

# Licence

Licenced under MIT. See [LICENCE.txt](LICENCE.txt).

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

