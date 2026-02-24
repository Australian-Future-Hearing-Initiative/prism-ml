# Introduction

AuditoryHuM: Auditory Scene Label Generation using Human-MLLM Collaboration.

* [Website](https://github.com/Australian-Future-Hearing-Initiative)
* [Paper](https://arxiv.org/abs/2602.19409)
* [Code](https://github.com/Australian-Future-Hearing-Initiative/prism-ml)
* [Supplementary Data](https://huggingface.co/datasets/hzhongresearch/auditoryhum_supplementary)
* [Demo Website With Alignment Samples](https://huggingface.co/spaces/hzhongresearch/auditoryhum_samples)

# Setup and run

```
# Setup
sudo apt update
sudo apt install --yes ffmpeg 7zip git git-lfs gzip python3 python3-pip python3-venv tar wget unzip
python3 -m venv env_cluster_analysis
source env_cluster_analysis/bin/activate
git clone https://github.com/Australian-Future-Hearing-Initiative/prism-ml.git
cd prism-ml/prism-ml-auditoryhum
pip install --upgrade pip
pip install --requirement requirements.txt

# Get TAU Urban Acoustic Scenes 2019 Development dataset
./tau2019.sh

# Alternatively get TAU Urban Audio Visual Scenes 2021 Development dataset
# ./tau2021.sh

# Alternatively get the AHEAD-DS dataset
# git clone https://huggingface.co/datasets/hzhongresearch/ahead_ds
# mv ahead_ds/*.wav .
# rm -rf ahead_ds

# Alternatively get the ADVANCE dataset
# mkdir advance
# wget https://zenodo.org/records/3828124/files/ADVANCE_sound.zip
# 7z e ADVANCE_sound.zip -o"advance"
# mv advance/*.wav .
# rm -rf advance

# Get models
git clone https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
git clone https://huggingface.co/sarulab-speech/human-clap-wsce-mse-mae
git clone https://huggingface.co/laion/clap-htsat-fused
git clone https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
git clone https://huggingface.co/Qwen/Qwen2.5-Omni-3B
# Fix Qwen2.5-Omni-3B broken tokenizer when using git clone
wget https://huggingface.co/Qwen/Qwen2.5-Omni-3B/resolve/main/tokenizer.json
mv tokenizer.json Qwen2.5-Omni-3B
git clone https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
git clone https://huggingface.co/unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit
git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Convert audio dataset to 16 kHz and 16-bit signed audio
python3 util_resample.py --filedir="*.wav" --sampling_rate=16000 --bits="s16" --channels=1

# Generate labels using various models
python3 label_qwen2a.py --filedir="*.wav" --qwen2a_model="Qwen2-Audio-7B-Instruct" --text_label_csv=qwen2a_text_labels.csv
python3 label_qwen2_5o.py --filedir="*.wav" --qwen2_5o_model="Qwen2.5-Omni-3B" --text_label_csv=qwen2_5o_text_labels.csv
python3 label_qwen3o.py --filedir="*.wav" --qwen3o_model="Qwen3-Omni-30B-A3B-Instruct" --text_label_csv=qwen3o_text_labels.csv
python3 label_gemma3n.py --filedir="*.wav" --gemma3n_model="gemma-3n-E2B-it-unsloth-bnb-4bit" --text_label_csv=gemma3n_text_labels.csv
python3 label_google_gen_ai_cloud.py --filedir="*.wav" --genai_id="gemini-2.5-flash" --api_key=<key> --text_label_csv=online_text_labels.csv

# Generate features and embeddings
python3 features_mfcc.py --filedir="*.wav" --mfcc_npy=mfcc.npy
python3 features_ast.py --filedir="*.wav" --ast_model="ast-finetuned-audioset-10-10-0.4593" --ast_npy=ast.npy
python3 features_qwen2a.py --filedir="*.wav" --qwen2a_model="Qwen2-Audio-7B-Instruct" --qwen2a_npy=qwen2a.npy
python3 features_gemma3n.py --filedir="*.wav" --gemma3n_model="gemma-3n-E2B-it-unsloth-bnb-4bit" --gemma3n_npy=gemma3n.npy
python3 features_clap.py --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=clap.npy

# Get top scoring labels
python3 features_clap_text.py --filedir="*.wav" --clap_model="human-clap-wsce-mse-mae" --clap_npy=clap.npy --text_label_csv=qwen2a_text_labels.csv --top_label_scores_csv=top_label_scores.csv
# Convert top scoring labels to text embeddings
python3 features_st.py --st_model="all-mpnet-base-v2" --label_csv=top_label_scores.csv --st_npy=st.npy

# Cluster features
python3 cluster_ag.py --cluster_npy=st.npy --n_clusters=5 --label_csv=st_ag.csv --png_plot=st_ag.png
python3 cluster_kmeans.py --cluster_npy=st.npy --n_clusters=3 --label_csv=st_kmeans.csv --png_plot=st_kmeans.png
python3 cluster_spectral.py --cluster_npy=st.npy --n_clusters=3 --label_csv=st_spectral.csv --png_plot=st_spectral.png
python3 cluster_hdbscan.py --cluster_npy=st.npy --min_cluster_size=5 --label_csv=st_hdbscan.csv --png_plot=st_hdbscan.png

# Composite label stats
python3 composite_label_stats.py --label_csv=st_ag.csv --top_label_scores_csv=top_label_scores.csv

# Check scores between MLLM vs MLLM + Human
python3 util_merge_csv.py --csv1=qwen2a_text_labels.csv --csv2=human_annotations.csv --csv3=top_label_scores_with_annotations.csv
python3 util_score_boost.py --csv1=top_label_scores.csv --csv2=top_label_scores_with_annotations.csv --csv3=human_annotations.csv

# To produce the results from the paper
# Recommended to run the following script manually one-line-at-a-time
# Replace human-clap-wsce-mse-mae to test alternative CLAP implementations
# Replace all-mpnet-base-v2 to test alternative Sentence-Transformers
./auditoryhum_tests.sh

# Initiate a MLLM prompt to enter values for composite labels
transformers chat --model_name_or_path="Qwen2.5-Omni-3B"
```

# Licence

Licenced under MIT. See [LICENCE.txt](LICENCE.txt).

Attribution.

```
@misc{zhong2026auditoryhumauditoryscenelabel,
      title={AuditoryHuM: Auditory Scene Label Generation and Clustering using Human-MLLM Collaboration}, 
      author={Henry Zhong and Jörg M. Buchholz and Julian Maclaren and Simon Carlile and Richard F. Lyon},
      year={2026},
      eprint={2602.19409},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2602.19409}, 
}
```
