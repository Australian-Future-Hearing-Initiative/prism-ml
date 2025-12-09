# PRISM-ML
Tools and scripts for AHEAD-DS data preparation and YAMNet+ (OpenYAMNet) training/inference.

## About
This repo supports the AHEAD-DS/YAMNet+ paper by providing:
- dataset build scripts for AHEAD-DS and AHEAD-DS-unmixed
- training pipeline for OpenYAMNet/YAMNet+ on those datasets
- legacy YAMNet conversion utilities

## Project Map
- `prism-ml-dataset-ahead-ds/`: build AHEAD-DS and AHEAD-DS-unmixed from raw sources (large downloads). See that README for full instructions and warnings about size/time.
- `prism-ml-yamnetp-tune/`: train and evaluate OpenYAMNet/YAMNet+ using AHEAD-DS, plus inference and TFLite conversion.
- `prism-ml-yamnet-legacy/`: convert the original YAMNet weights to TF SavedModel format compatible with TF 2.16+.
- Other datasets/examples: `prism-ml-dataset-audioset/`, `prism-ml-dataset-pace/`, `prism-ml-yamnet-demo/`, etc.

## Quick Start
Common setup (per subproject, adapt paths as needed):
```bash
python3 -m venv env_prism
source env_prism/bin/activate
pip install --upgrade pip
```

Then follow the relevant README:
- Dataset build: `prism-ml-dataset-ahead-ds/README.md`
- Model training/inference: `prism-ml-yamnetp-tune/README.md`
- Legacy YAMNet conversion: `prism-ml-yamnet-legacy/README.md`

## Data and Models
- AHEAD-DS: https://huggingface.co/datasets/hzhongresearch/ahead_ds
- AHEAD-DS (unmixed): https://huggingface.co/datasets/hzhongresearch/ahead_ds_unmixed
- OpenYAMNet/YAMNet+ models: https://huggingface.co/hzhongresearch/yamnetp_ahead_ds

## Licence
MIT (see `LICENCE.txt` in each subproject).

## Contact
For questions, feedback, or collaboration opportunities, please contact:
- **Name:** Romaric Bouveret
- **Email:** [romaric.bouveret@mq.edu.au](mailto\:romaric.bouveret@mq.edu.au)
