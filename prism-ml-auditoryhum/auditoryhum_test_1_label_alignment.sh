# Get the human provided labels
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
python3 label_gemma3n.py --filedir="*.wav" --gemma3n_model="gemma-3n-E2B-it-unsloth-bnb-4bit" --text_label_csv=advance_gemma3n_e2b_labels.csv
python3 label_qwen2a.py --filedir="*.wav" --qwen2a_model="Qwen2-Audio-7B-Instruct" --text_label_csv=advance_qwen2a_labels.csv
python3 label_qwen2_5o.py --filedir="*.wav" --qwen2_5o_model="Qwen2.5-Omni-7B" --text_label_csv=advance_qwen2_5o_labels.csv

# Merge with annotations
python3 util_merge_csv.py --csv1=advance_gemma3n_e2b_labels.csv --csv2=advance_gemma3n_e2b_labels_annotations.csv --csv3=advance_gemma3n_e2b_labels_combined.csv
python3 util_merge_csv.py --csv1=advance_qwen2a_labels.csv --csv2=advance_qwen2a_labels_annotations.csv --csv3=advance_qwen2a_labels_combined.csv
python3 util_merge_csv.py --csv1=advance_qwen2_5o_labels.csv --csv2=advance_qwen2_5o_labels_annotations.csv --csv3=advance_qwen2_5o_labels_combined.csv

# Get features for labels
python3 features_clap.py --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=advance_clap.npy
# Get metrics for labels
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=advance_clap.npy --text_label_csv=advance_gemma3n_e2b_labels.csv --top_label_scores_csv=advance_gemma3n_e2b_labels_scores.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=advance_clap.npy --text_label_csv=advance_qwen2a_labels.csv --top_label_scores_csv=advance_qwen2a_labels_scores.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=advance_clap.npy --text_label_csv=advance_qwen2_5o_labels.csv --top_label_scores_csv=advance_qwen2_5o_labels_scores.csv
# Get metrics with annotations
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=advance_clap.npy --text_label_csv=advance_gemma3n_e2b_labels_combined.csv --top_label_scores_csv=advance_gemma3n_e2b_labels_combined_scores.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=advance_clap.npy --text_label_csv=advance_qwen2a_labels_combined.csv --top_label_scores_csv=advance_qwen2a_labels_combined_scores.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=advance_clap.npy --text_label_csv=advance_qwen2_5o_labels_combined.csv --top_label_scores_csv=advance_qwen2_5o_labels_combined_scores.csv
# Compare metrics, MLLM vs Human + MLLM
python3 util_score_boost.py --csv1=advance_gemma3n_e2b_labels_scores.csv --csv2=advance_gemma3n_e2b_labels_combined_scores.csv --csv3=advance_gemma3n_e2b_labels_annotations.csv
python3 util_score_boost.py --csv1=advance_qwen2a_labels_scores.csv --csv2=advance_qwen2a_labels_combined_scores.csv --csv3=advance_qwen2a_labels_annotations.csv
python3 util_score_boost.py --csv1=advance_qwen2_5o_labels_scores.csv --csv2=advance_qwen2_5o_labels_combined_scores.csv --csv3=advance_qwen2_5o_labels_annotations.csv

# Human labelling strategy tests
python3 util_merge_csv.py --csv1=advance_qwen2_5o_labels.csv --csv2=advance_qwen2_5o_human_strat.csv --csv3=advance_qwen2_5o_labels_human_strat.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=advance_clap.npy --text_label_csv=advance_qwen2_5o_labels_human_strat.csv --top_label_scores_csv=advance_qwen2_5o_human_strat_scores.csv

# AHEAD-DS
rm *.wav
git clone https://huggingface.co/datasets/hzhongresearch/ahead_ds
mv ahead_ds/*.wav .
rm -rf ahead_ds

# Generate labels from scratch, otherwise use provided labels
python3 label_gemma3n.py --filedir="*.wav" --gemma3n_model="gemma-3n-E2B-it-unsloth-bnb-4bit" --text_label_csv=ahead-ds_gemma3n_e2b_labels.csv
python3 label_qwen2a.py --filedir="*.wav" --qwen2a_model="Qwen2-Audio-7B-Instruct" --text_label_csv=ahead-ds_qwen2a_labels.csv
python3 label_qwen2_5o.py --filedir="*.wav" --qwen2_5o_model="Qwen2.5-Omni-7B" --text_label_csv=ahead-ds_qwen2_5o_labels.csv

# Merge with annotations
python3 util_merge_csv.py --csv1=ahead-ds_gemma3n_e2b_labels.csv --csv2=ahead-ds_gemma3n_e2b_labels_annotations.csv --csv3=ahead-ds_gemma3n_e2b_labels_combined.csv
python3 util_merge_csv.py --csv1=ahead-ds_qwen2a_labels.csv --csv2=ahead-ds_qwen2a_labels_annotations.csv --csv3=ahead-ds_qwen2a_labels_combined.csv
python3 util_merge_csv.py --csv1=ahead-ds_qwen2_5o_labels.csv --csv2=ahead-ds_qwen2_5o_labels_annotations.csv --csv3=ahead-ds_qwen2_5o_labels_combined.csv

# Get features for labels
python3 features_clap.py --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=ahead-ds_clap.npy
# Get metrics for labels
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=ahead-ds_clap.npy --text_label_csv=ahead-ds_gemma3n_e2b_labels.csv --top_label_scores_csv=ahead-ds_gemma3n_e2b_labels_scores.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=ahead-ds_clap.npy --text_label_csv=ahead-ds_qwen2a_labels.csv --top_label_scores_csv=ahead-ds_qwen2a_labels_scores.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=ahead-ds_clap.npy --text_label_csv=ahead-ds_qwen2_5o_labels.csv --top_label_scores_csv=ahead-ds_qwen2_5o_labels_scores.csv
# Get metrics with annotations
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=ahead-ds_clap.npy --text_label_csv=ahead-ds_gemma3n_e2b_labels_combined.csv --top_label_scores_csv=ahead-ds_gemma3n_e2b_labels_combined_scores.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=ahead-ds_clap.npy --text_label_csv=ahead-ds_qwen2a_labels_combined.csv --top_label_scores_csv=ahead-ds_qwen2a_labels_combined_scores.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=ahead-ds_clap.npy --text_label_csv=ahead-ds_qwen2_5o_labels_combined.csv --top_label_scores_csv=ahead-ds_qwen2_5o_labels_combined_scores.csv
# Compare metrics, MLLM vs Human + MLLM
python3 util_score_boost.py --csv1=ahead-ds_gemma3n_e2b_labels_scores.csv --csv2=ahead-ds_gemma3n_e2b_labels_combined_scores.csv --csv3=ahead-ds_gemma3n_e2b_labels_annotations.csv
python3 util_score_boost.py --csv1=ahead-ds_qwen2a_labels_scores.csv --csv2=ahead-ds_qwen2a_labels_combined_scores.csv --csv3=ahead-ds_qwen2a_labels_annotations.csv
python3 util_score_boost.py --csv1=ahead-ds_qwen2_5o_labels_scores.csv --csv2=ahead-ds_qwen2_5o_labels_combined_scores.csv --csv3=ahead-ds_qwen2_5o_labels_annotations.csv

# TAU 2019
rm *.wav
./tau2019.sh
python3 util_resample.py --filedir="*.wav" --sampling_rate=16000 --bits="s16" --channels=1

# Generate labels from scratch, otherwise use provided labels
python3 label_gemma3n.py --filedir="*.wav" --gemma3n_model="gemma-3n-E2B-it-unsloth-bnb-4bit" --text_label_csv=tau2019_gemma3n_e2b_labels.csv
python3 label_qwen2a.py --filedir="*.wav" --qwen2a_model="Qwen2-Audio-7B-Instruct" --text_label_csv=tau2019_qwen2a_labels.csv
python3 label_qwen2_5o.py --filedir="*.wav" --qwen2_5o_model="Qwen2.5-Omni-7B" --text_label_csv=tau2019_qwen2_5o_labels.csv

# Merge with annotations
python3 util_merge_csv.py --csv1=tau2019_gemma3n_e2b_labels.csv --csv2=tau2019_gemma3n_e2b_labels_annotations.csv --csv3=tau2019_gemma3n_e2b_labels_combined.csv
python3 util_merge_csv.py --csv1=tau2019_qwen2a_labels.csv --csv2=tau2019_qwen2a_labels_annotations.csv --csv3=tau2019_qwen2a_labels_combined.csv
python3 util_merge_csv.py --csv1=tau2019_qwen2_5o_labels.csv --csv2=tau2019_qwen2_5o_labels_annotations.csv --csv3=tau2019_qwen2_5o_labels_combined.csv

# Get features for labels
python3 features_clap.py --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=tau2019_clap.npy
# Get metrics for labels
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=tau2019_clap.npy --text_label_csv=tau2019_gemma3n_e2b_labels.csv --top_label_scores_csv=tau2019_gemma3n_e2b_labels_scores.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=tau2019_clap.npy --text_label_csv=tau2019_qwen2a_labels.csv --top_label_scores_csv=tau2019_qwen2a_labels_scores.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=tau2019_clap.npy --text_label_csv=tau2019_qwen2_5o_labels.csv --top_label_scores_csv=tau2019_qwen2_5o_labels_scores.csv
# Get metrics with annotations
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=tau2019_clap.npy --text_label_csv=tau2019_gemma3n_e2b_labels_combined.csv --top_label_scores_csv=tau2019_gemma3n_e2b_labels_combined_scores.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=tau2019_clap.npy --text_label_csv=tau2019_qwen2a_labels_combined.csv --top_label_scores_csv=tau2019_qwen2a_labels_combined_scores.csv
python3 features_clap_text.py --label_limit=2 --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=tau2019_clap.npy --text_label_csv=tau2019_qwen2_5o_labels_combined.csv --top_label_scores_csv=tau2019_qwen2_5o_labels_combined_scores.csv
# Compare metrics, MLLM vs Human + MLLM
python3 util_score_boost.py --csv1=tau2019_gemma3n_e2b_labels_scores.csv --csv2=tau2019_gemma3n_e2b_labels_combined_scores.csv --csv3=tau2019_gemma3n_e2b_labels_annotations.csv
python3 util_score_boost.py --csv1=tau2019_qwen2a_labels_scores.csv --csv2=tau2019_qwen2a_labels_combined_scores.csv --csv3=tau2019_qwen2a_labels_annotations.csv
python3 util_score_boost.py --csv1=tau2019_qwen2_5o_labels_scores.csv --csv2=tau2019_qwen2_5o_labels_combined_scores.csv --csv3=tau2019_qwen2_5o_labels_annotations.csv
