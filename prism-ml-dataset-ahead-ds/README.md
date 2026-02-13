# Introduction
Scripts to download several datasets and merge them into Another HEAring aiD scenes DataSet (AHEAD-DS).

* [Website](https://github.com/Australian-Future-Hearing-Initiative)
* [Paper](https://arxiv.org/abs/2508.10360)
* [Code](https://github.com/Australian-Future-Hearing-Initiative/prism-ml/prism-ml-yamnetp-tune)
* [Dataset AHEAD-DS](https://huggingface.co/datasets/hzhongresearch/ahead_ds)
* [Dataset AHEAD-DS unmixed](https://huggingface.co/datasets/hzhongresearch/ahead_ds_unmixed)
* [Models](https://huggingface.co/hzhongresearch/yamnetp_ahead_ds)

# Quick Start (download prebuilt)
- To **use the data**, download from Hugging Face (links above). Each repo contains the ZIP and CSVs; no processing needed.
- To **rebuild from raw audio**, follow the steps below. This is heavy (hundreds of GB downloaded and processed) and time-consuming.

# Setup and download
```
# Setup 
sudo apt update
sudo apt install --yes bzip2 ffmpeg git gzip python3 tar wget zip
python3 -m venv env_ahead
source env_ahead/bin/activate
git clone https://github.com/Australian-Future-Hearing-Initiative/prism-ml.git
cd prism-ml/prism-ml-dataset-ahead-ds
pip install --upgrade pip
pip install --requirement requirements.txt

# Get HEAR-DS
# https://www.hz-ol.de/en/hear-ds.html
wget -O CocktailParty.tar.bz2 https://download.hz-ol.de/hear-ds-data/HEAR-DS/AudioSnippets/AudioSnippets-ITC-16kHz/CocktailParty.tar.bz2
wget -O InTraffic.tar.bz2 https://download.hz-ol.de/hear-ds-data/HEAR-DS/AudioSnippets/AudioSnippets-ITC-16kHz/InTraffic.tar.bz2
wget -O InVehicle.tar.bz2 https://download.hz-ol.de/hear-ds-data/HEAR-DS/AudioSnippets/AudioSnippets-ITC-16kHz/InVehicle.tar.bz2
wget -O Music.tar.bz2 https://download.hz-ol.de/hear-ds-data/HEAR-DS/AudioSnippets/AudioSnippets-ITC-16kHz/Music.tar.bz2
wget -O QuietIndoors.tar.bz2 https://download.hz-ol.de/hear-ds-data/HEAR-DS/AudioSnippets/AudioSnippets-ITC-16kHz/QuietIndoors.tar.bz2
wget -O ReverberantEnvironment.tar.bz2 https://download.hz-ol.de/hear-ds-data/HEAR-DS/AudioSnippets/AudioSnippets-ITC-16kHz/ReverberantEnvironment.tar.bz2
wget -O WindTurbulence.tar.bz2 https://download.hz-ol.de/hear-ds-data/HEAR-DS/AudioSnippets/AudioSnippets-ITC-16kHz/WindTurbulence.tar.bz2

# Get CHiME6 Dev
# https://openslr.org/150/
wget -O CHiME6_dev.tar.gz https://www.openslr.org/resources/150/CHiME6_dev.tar.gz

# Extract datasets
tar xvf CocktailParty.tar.bz2
tar xvf InTraffic.tar.bz2
tar xvf InVehicle.tar.bz2
tar xvf Music.tar.bz2
tar xvf QuietIndoors.tar.bz2
tar xvf ReverberantEnvironment.tar.bz2
tar xvf WindTurbulence.tar.bz2
tar xvf CHiME6_dev.tar.gz

#### Create directories

# Create directories for environment sounds
mkdir cocktail_party
mkdir interfering_speakers
mkdir in_traffic
mkdir in_vehicle
mkdir music_snippet
mkdir quiet_indoors
mkdir reverberant_environment
mkdir wind_turbulence

# Create directories for speech
mkdir chime_speech

# Create directories for environment sounds to be mixed with speech
mkdir in_traffic_env
mkdir in_vehicle_env
mkdir music_env
mkdir quiet_indoors_env
mkdir reverberant_environment_env
mkdir wind_turbulence_env

# Create directories for speech to be mixed with environment sounds
mkdir in_traffic_speech
mkdir in_vehicle_speech
mkdir music_speech
mkdir quiet_indoors_speech
mkdir reverberant_environment_speech
mkdir wind_turbulence_speech

# Create directories for speech in environment sounds
mkdir speech_in_traffic
mkdir speech_in_vehicle
mkdir speech_in_music
mkdir speech_in_quiet_indoors
mkdir speech_in_reverberant_environment
mkdir speech_in_wind_turbulence

# Create directory to store final datasets
mkdir ahead_ds
mkdir ahead_ds_unmixed

#### Process sounds

# Resample to ensure all sounds are 16bit, 16000Hz, 1 channel
python3 ahead_ds_resample.py "CocktailParty/Background/*.wav" 16000 1
python3 ahead_ds_resample.py "InTraffic/Background/*.wav" 16000 1
python3 ahead_ds_resample.py "InVehicle/Background/*.wav" 16000 1
python3 ahead_ds_resample.py "Music/Background/*.wav" 16000 1
python3 ahead_ds_resample.py "QuietIndoors/Background/*.wav" 16000 1
python3 ahead_ds_resample.py "ReverberantEnvironment/Background/*.wav" 16000 1
python3 ahead_ds_resample.py "WindTurbulence/Background/*.wav" 16000 1
python3 ahead_ds_resample.py "CHiME6_dev/CHiME6/audio/dev/*.wav" 16000 1

# Analyse sound for value of root mean squared (RMS)
# This can be useful for standardisation
# CHiME 6 Dev RMS 2607.63
# HEAR-DS Music RMS 3429.22
# HEAR-DS CocktailParty RMS 97.55
# HEAR-DS InTraffic RMS 104.47
# HEAR-DS InVehicle RMS 272.49
# HEAR-DS QuietIndoors RMS 5.92
# HEAR-DS ReverberantEnvironment RMS 97.13
# HEAR-DS WindTurbulence RMS 468.63
python3 ahead_ds_analyse.py "CHiME6_dev/CHiME6/audio/dev/S*CH*.wav"
python3 ahead_ds_analyse.py "Music/Background/*.wav"
python3 ahead_ds_analyse.py "CocktailParty/Background/*.wav"
python3 ahead_ds_analyse.py "InTraffic/Background/*.wav"
python3 ahead_ds_analyse.py "InVehicle/Background/*.wav"
python3 ahead_ds_analyse.py "QuietIndoors/Background/*.wav"
python3 ahead_ds_analyse.py "ReverberantEnvironment/Background/*.wav"
python3 ahead_ds_analyse.py "WindTurbulence/Background/*.wav"

# Scale datasets based on RMS
# CHiME 6 Dev, HEAR-DS Music and HEAR-DS excluding Music were recorded using 3 hardware configurations respectively
# Adjust levels of CocktailParty and Music to be similar to CHiME 6 Dev
# All other labels are boosted by the same amount as CocktailParty
python3 ahead_ds_rescale.py "Music/Background/*.wav" 34.0 28.0
python3 ahead_ds_rescale.py "CocktailParty/Background/*.wav" 1.0 28.0
python3 ahead_ds_rescale.py "InTraffic/Background/*.wav" 1.0 28.0
python3 ahead_ds_rescale.py "InVehicle/Background/*.wav" 1.0 28.0
python3 ahead_ds_rescale.py "QuietIndoors/Background/*.wav" 1.0 28.0
python3 ahead_ds_rescale.py "ReverberantEnvironment/Background/*.wav" 1.0 28.0
python3 ahead_ds_rescale.py "WindTurbulence/Background/*.wav" 1.0 28.0

# Extract snippets of speech from CHiME6 Dev
python3 ahead_ds_cut_sound.py "CHiME6_dev/CHiME6/audio/dev/S*CH*.wav" "chime_speech/speech_{:05d}.wav" 16000 17616000 160000

# Move and rename cocktail party sounds
python3 ahead_ds_move_rename.py "CocktailParty/Background/*.wav" "cocktail_party/cocktail_party_{:05d}.wav"

# Split data between environment and mixing sounds
python3 ahead_ds_split_rename.py "InTraffic/Background/*.wav" "in_traffic/in_traffic_{:05d}.wav" "in_traffic_env/in_traffic_env_{:05d}.wav"
python3 ahead_ds_split_rename.py "InVehicle/Background/*.wav" "in_vehicle/in_vehicle_{:05d}.wav" "in_vehicle_env/in_vehicle_env_{:05d}.wav"
python3 ahead_ds_split_rename.py "Music/Background/*.wav" "music_snippet/music_{:05d}.wav" "music_env/music_env_{:05d}.wav"
python3 ahead_ds_split_rename.py "QuietIndoors/Background/*.wav" "quiet_indoors/quiet_indoors_{:05d}.wav" "quiet_indoors_env/quiet_indoors_env_{:05d}.wav"
python3 ahead_ds_split_rename.py "ReverberantEnvironment/Background/*.wav" "reverberant_environment/reverberant_environment_{:05d}.wav" "reverberant_environment_env/reverberant_environment_env_{:05d}.wav"
python3 ahead_ds_split_rename.py "WindTurbulence/Background/*.wav" "wind_turbulence/wind_turbulence_{:05d}.wav" "wind_turbulence_env/wind_turbulence_env_{:05d}.wav"

# Split speech between samples needed for mixing with environment and interfering_speakers class
python3 ahead_ds_divide_speech.py "chime_speech/*.wav" "in_traffic_env/*.wav" "in_vehicle_env/*.wav" "music_env/*.wav" "quiet_indoors_env/*.wav" "reverberant_environment_env/*.wav" "wind_turbulence_env/*.wav" "in_traffic_speech/in_traffic_speech_{:05d}.wav" "in_vehicle_speech/in_vehicle_speech_{:05d}.wav" "music_speech/music_speech_{:05d}.wav" "quiet_indoors_speech/quiet_indoors_speech_{:05d}.wav" "reverberant_environment_speech/reverberant_environment_speech_{:05d}.wav" "wind_turbulence_speech/wind_turbulence_speech_{:05d}.wav" "interfering_speakers/interfering_speakers_{:05d}.wav"

# Mix speech and environment sounds
python3 ahead_ds_mix.py "in_traffic_env/*.wav" "in_traffic_speech/*.wav" "speech_in_traffic/speech_in_traffic_{:05d}.wav" "-10, -5, 0, 5, 10"
python3 ahead_ds_mix.py "in_vehicle_env/*.wav" "in_vehicle_speech/*.wav" "speech_in_vehicle/speech_in_vehicle_{:05d}.wav" "-10, -5, 0, 5, 10"
python3 ahead_ds_mix.py "music_env/*.wav" "music_speech/*.wav" "speech_in_music/speech_in_music_{:05d}.wav" "-10, -5, 0, 5, 10"
python3 ahead_ds_mix.py "quiet_indoors_env/*.wav" "quiet_indoors_speech/*.wav" "speech_in_quiet_indoors/speech_in_quiet_indoors_{:05d}.wav" "-10, -5, 0, 5, 10"
python3 ahead_ds_mix.py "reverberant_environment_env/*.wav" "reverberant_environment_speech/*.wav" "speech_in_reverberant_environment/speech_in_reverberant_environment_{:05d}.wav" "-10, -5, 0, 5, 10"
python3 ahead_ds_mix.py "wind_turbulence_env/*.wav" "wind_turbulence_speech/*.wav" "speech_in_wind_turbulence/speech_in_wind_turbulence_{:05d}.wav" "-10, -5, 0, 5, 10"

# Generate the CSVs specifying the dataset
python3 ahead_ds_csv.py "cocktail_party/*.wav" "interfering_speakers/*.wav" "in_traffic/*.wav" "in_vehicle/*.wav" "music_snippet/*.wav" "quiet_indoors/*.wav" "reverberant_environment/*.wav" "wind_turbulence/*.wav" "speech_in_traffic/*.wav" "speech_in_vehicle/*.wav" "speech_in_music/*.wav" "speech_in_quiet_indoors/*.wav" "speech_in_reverberant_environment/*.wav" "speech_in_wind_turbulence/*.wav" "cocktail_party" "interfering_speakers" "in_traffic" "in_vehicle" "music" "quiet_indoors" "reverberant_environment" "wind_turbulence" "speech_in_traffic" "speech_in_vehicle" "speech_in_music" "speech_in_quiet_indoors" "speech_in_reverberant_environment" "speech_in_wind_turbulence" "ahead_ds_all.csv" "ahead_ds_training.csv" "ahead_ds_validation.csv" "ahead_ds_testing.csv" "0, 0, 1, 0, 0, 2, 0, 0, 2, 0"

#### Process unmixed sounds

python3 ahead_ds_unmixed_csv.py "cocktail_party/*.wav" "interfering_speakers/*.wav" "in_traffic/*.wav" "in_vehicle/*.wav" "music_snippet/*.wav" "quiet_indoors/*.wav" "reverberant_environment/*.wav" "wind_turbulence/*.wav" "in_traffic_env/*.wav" "in_vehicle_env/*.wav" "music_env/*.wav" "quiet_indoors_env/*.wav" "reverberant_environment_env/*.wav" "wind_turbulence_env/*.wav" "in_traffic_speech/*.wav" "in_vehicle_speech/*.wav" "music_speech/*.wav" "quiet_indoors_speech/*.wav" "reverberant_environment_speech/*.wav" "wind_turbulence_speech/*.wav" "0, 0, 1, 0, 0, 2, 0, 0, 2, 0" "-10, -5, 0, 5, 10" "ahead_ds_unmixed.csv"

#### Package files

mv ahead_ds_all.csv ahead_ds/
mv ahead_ds_training.csv ahead_ds/
mv ahead_ds_validation.csv ahead_ds/
mv ahead_ds_testing.csv ahead_ds/

cp cocktail_party/*.wav ahead_ds/
cp interfering_speakers/*.wav ahead_ds/
cp in_traffic/*.wav ahead_ds/
cp in_vehicle/*.wav ahead_ds/
cp music_snippet/*.wav ahead_ds/
cp quiet_indoors/*.wav ahead_ds/
cp reverberant_environment/*.wav ahead_ds/
cp wind_turbulence/*.wav ahead_ds/
mv speech_in_traffic/*.wav ahead_ds/
mv speech_in_vehicle/*.wav ahead_ds/
mv speech_in_music/*.wav ahead_ds/
mv speech_in_quiet_indoors/*.wav ahead_ds/
mv speech_in_reverberant_environment/*.wav ahead_ds/
mv speech_in_wind_turbulence/*.wav ahead_ds/

cp ahead_ds_licence.txt ahead_ds/LICENCE.txt
cp ahead_ds_readme.txt ahead_ds/README.md

zip -r ahead_ds.zip ahead_ds

# Analyse AHEAD-DS RMS
python3 ahead_ds_analyse.py "ahead_ds/*.wav"

#### Package unmixed files

mv ahead_ds_unmixed.csv ahead_ds_unmixed/

cp ahead_ds_licence.txt ahead_ds_unmixed/LICENCE.txt
cp ahead_ds_unmixed_readme.txt ahead_ds_unmixed/README.md

mv cocktail_party ahead_ds_unmixed/
mv interfering_speakers ahead_ds_unmixed/
mv in_traffic ahead_ds_unmixed/
mv in_vehicle ahead_ds_unmixed/
mv music_snippet ahead_ds_unmixed/
mv quiet_indoors ahead_ds_unmixed/
mv reverberant_environment ahead_ds_unmixed/
mv wind_turbulence ahead_ds_unmixed/

mv in_traffic_env ahead_ds_unmixed/
mv in_vehicle_env ahead_ds_unmixed/
mv music_env ahead_ds_unmixed/
mv quiet_indoors_env ahead_ds_unmixed/
mv reverberant_environment_env ahead_ds_unmixed/
mv wind_turbulence_env ahead_ds_unmixed/

mv in_traffic_speech ahead_ds_unmixed/
mv in_vehicle_speech ahead_ds_unmixed/
mv music_speech ahead_ds_unmixed/
mv quiet_indoors_speech ahead_ds_unmixed/
mv reverberant_environment_speech ahead_ds_unmixed/
mv wind_turbulence_speech ahead_ds_unmixed/

zip -r ahead_ds_unmixed.zip ahead_ds_unmixed

#### Cleanup

rm -r CocktailParty
rm -r InTraffic
rm -r InVehicle
rm -r Music
rm -r QuietIndoors
rm -r ReverberantEnvironment
rm -r WindTurbulence

rm -r CHiME6_dev
rm -r chime_speech

#rm -r cocktail_party
#rm -r interfering_speakers
#rm -r in_traffic
#rm -r in_vehicle
#rm -r music_snippet
#rm -r quiet_indoors
#rm -r reverberant_environment
#rm -r wind_turbulence

#rm -r in_traffic_env
#rm -r in_vehicle_env
#rm -r music_env
#rm -r quiet_indoors_env
#rm -r reverberant_environment_env
#rm -r wind_turbulence_env

#rm -r in_traffic_speech
#rm -r in_vehicle_speech
#rm -r music_speech
#rm -r quiet_indoors_speech
#rm -r reverberant_environment_speech
#rm -r wind_turbulence_speech

rm -r speech_in_traffic
rm -r speech_in_vehicle
rm -r speech_in_music
rm -r speech_in_quiet_indoors
rm -r speech_in_reverberant_environment
rm -r speech_in_wind_turbulence

rm -r ahead_ds
rm -r ahead_ds_unmixed
```

# Licence
Licenced under MIT. See [LICENCE.txt](LICENCE.txt).

* ahead_ds_licence.txt is the licence for the data of AHEAD-DS and AHEAD-DS unmixed.
* LICENCE.txt is the licence for the code in this directory.

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
