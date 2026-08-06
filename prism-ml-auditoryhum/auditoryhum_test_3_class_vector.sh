# Get the provided labels
git clone https://huggingface.co/datasets/hzhongresearch/auditoryhum_supplementary
mv auditoryhum_supplementary/*.csv .
rm -rf auditoryhum_supplementary

# ADVANCE
rm *.wav
wget https://zenodo.org/records/3828124/files/ADVANCE_sound.zip
7z e ADVANCE_sound.zip -o"advance"
mv advance/*.wav .
rm -rf advance
python3 util_resample.py --filedir="*.wav" --sampling_rate=16000 --bits="s16" --channels=1

# Generate labels from scratch, otherwise use provided labels
python3 label_qwen2_5o.py --filedir="*.wav" --qwen2_5o_model="Qwen2.5-Omni-7B" --text_label_csv=advance_qwen2_5o_labels.csv
# Get features for labels
python3 features_clap.py --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=advance_clap.npy
# Get metrics for labels
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=advance_clap.npy --text_label_csv=advance_qwen2_5o_labels.csv --top_label_scores_csv=advance_qwen2_5o_labels_scores.csv
# Get class vector features
python3 features_cvector.py --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --scores_csv=advance_qwen2_5o_labels_scores.csv --cvector_npy=advance_cvector.npy

# AHEAD-DS
rm *.wav
git clone https://huggingface.co/datasets/hzhongresearch/ahead_ds
mv ahead_ds/*.wav .
rm -rf ahead_ds

# Generate labels from scratch, otherwise use provided labels
python3 label_qwen2_5o.py --filedir="*.wav" --qwen2_5o_model="Qwen2.5-Omni-7B" --text_label_csv=ahead-ds_qwen2_5o_labels.csv
# Get features for labels
python3 features_clap.py --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=ahead-ds_clap.npy
# Get metrics for labels
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=ahead-ds_clap.npy --text_label_csv=ahead-ds_qwen2_5o_labels.csv --top_label_scores_csv=ahead-ds_qwen2_5o_labels_scores.csv
# Get class vector features
python3 features_cvector.py --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --scores_csv=ahead-ds_qwen2_5o_labels_scores.csv --cvector_npy=ahead-ds_cvector.npy

# TAU 2019
rm *.wav
./tau2019.sh
python3 util_resample.py --filedir="*.wav" --sampling_rate=16000 --bits="s16" --channels=1

# Generate labels from scratch, otherwise use provided labels
python3 label_qwen2_5o.py --filedir="*.wav" --qwen2_5o_model="Qwen2.5-Omni-7B" --text_label_csv=tau2019_qwen2_5o_labels.csv
# Get features for labels
python3 features_clap.py --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=tau2019_clap.npy
# Get metrics for labels
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=tau2019_clap.npy --text_label_csv=tau2019_qwen2_5o_labels.csv --top_label_scores_csv=tau2019_qwen2_5o_labels_scores.csv
# Get class vector features
python3 features_cvector.py --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --scores_csv=tau2019_qwen2_5o_labels_scores.csv --cvector_npy=tau2019_cvector.npy
