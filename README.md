# PRISM-ML
Tools and scripts for AuditoryHuM auditory scene clustering, AHEAD-DS data preparation and OpenYAMNet(YAMNet+) training/inference.

## About
This repo supports the AuditoryHuM/AHEAD-DS/OpenYAMNet papers by providing:
- clustering pipline for AuditoryHuM
- dataset build scripts for AHEAD-DS and AHEAD-DS-unmixed
- training pipeline for OpenYAMNet/YAMNet+ on those datasets
- legacy YAMNet conversion utilities

## Project Map
- `prism-ml-auditoryhum/`: auditory scene label discovery and clustering using MLLM label generation and zero-shot learning.
- `prism-ml-dataset-ahead-ds/`: build AHEAD-DS and AHEAD-DS-unmixed from raw sources (large downloads). See that README for full instructions and warnings about size/time.
- `prism-ml-yamnetp-tune/`: train and evaluate OpenYAMNet/YAMNet+ using AHEAD-DS, plus inference and TFLite conversion.
- `prism-ml-yamnet-legacy/`: convert the original YAMNet weights to TF SavedModel format compatible with TF 2.16+.

## Quick Start
Follow the relevant README:
- Auditory scene clustering: `prism-ml-auditoryhum/README.md`
- Dataset build: `prism-ml-dataset-ahead-ds/README.md`
- Model training/inference: `prism-ml-yamnetp-tune/README.md`
- Legacy YAMNet conversion: `prism-ml-yamnet-legacy/README.md`

## Paper, Data and Models
- AuditoryHuM paper: https://arxiv.org/abs/2602.19409
- AuditoryHuM supplementary data: https://huggingface.co/datasets/hzhongresearch/auditoryhum_supplementary
- AuditoryHuM demo website with alignment samples: https://huggingface.co/spaces/hzhongresearch/auditoryhum_samples
- AHEAD-DS and OpenYAMNet paper: https://arxiv.org/abs/2508.10360
- AHEAD-DS: https://huggingface.co/datasets/hzhongresearch/ahead_ds
- AHEAD-DS (unmixed): https://huggingface.co/datasets/hzhongresearch/ahead_ds_unmixed
- OpenYAMNet/YAMNet+ models: https://huggingface.co/hzhongresearch/yamnetp_ahead_ds

## Licence
See `LICENCE.txt` in each subproject.

## Contact
For questions, feedback, or collaboration opportunities, please contact:
- **Name:** Romaric Bouveret
- **Email:** [romaric.bouveret@mq.edu.au](mailto:romaric.bouveret@mq.edu.au)
