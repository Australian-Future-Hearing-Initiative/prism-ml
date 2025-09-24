---
language:
- en
license: cc-by-sa-4.0
tags:
- audio
task_categories:
- audio-classification
---

# Another HEaring AiD DataSet (AHEAD-DS)
Another HEaring AiD DataSet (AHEAD-DS) is an audio dataset labelled with audiologically relevant scene categories for hearing aids.

* [Website](https://github.com/Australian-Future-Hearing-Initiative)
* [Paper](https://arxiv.org/abs/2508.10360)
* [Code](https://github.com/Australian-Future-Hearing-Initiative/prism-ml/prism-ml-yamnetp-tune)
* [Dataset AHEAD-DS](https://huggingface.co/datasets/hzhongresearch/ahead_ds)
* [Dataset AHEAD-DS unmixed](https://huggingface.co/datasets/hzhongresearch/ahead_ds_unmixed)
* [Models](https://huggingface.co/hzhongresearch/yamnetp_ahead_ds)

## Description of data
All files are encoded as single channel WAV, 16 bit signed, sampled at 16 kHz with 10 seconds per recording.

| Category                          | Training | Validation | Testing | All  |
|:----------------------------------|:---------|:-----------|:--------|:-----|
| cocktail_party                    | 934      | 134        | 266     | 1334 |
| interfering_speakers              | 733      | 105        | 209     | 1047 |
| in_traffic                        | 370      | 53         | 105     | 528  |
| in_vehicle                        | 409      | 59         | 116     | 584  |
| music                             | 1047     | 150        | 299     | 1496 |
| quiet_indoors                     | 368      | 53         | 104     | 525  |
| reverberant_environment           | 156      | 22         | 44      | 222  |
| wind_turbulence                   | 307      | 44         | 88      | 439  |
| speech_in_traffic                 | 370      | 53         | 105     | 528  |
| speech_in_vehicle                 | 409      | 59         | 116     | 584  |
| speech_in_music                   | 1047     | 150        | 299     | 1496 |
| speech_in_quiet_indoors           | 368      | 53         | 104     | 525  |
| speech_in_reverberant_environment | 155      | 22         | 44      | 221  |
| speech_in_wind_turbulence         | 307      | 44         | 88      | 439  |
| Total                             | 6980     | 1001       | 1987    | 9968 |

# Licence
Licenced under CC BY-SA 4.0. See [LICENCE.txt](LICENCE.txt).

AHEAD-DS was derived from [HEAR-DS](https://www.hz-ol.de/en/hear-ds.html) (CC0 licence) and [CHiME 6 dev](https://openslr.org/150/) (CC BY-SA 4.0 licence). If you use this work, please cite the following publications.

Attribution.

```
@article{zhong2025dataset,
  title={A dataset and model for recognition of audiologically relevant environments for hearing aids: AHEAD-DS and YAMNet+},
  author={Zhong, Henry and Buchholz, J{\"o}rg M and Maclaren, Julian and Carlile, Simon and Lyon, Richard},
  journal={arXiv preprint arXiv:2508.10360},
  year={2025}
}
```

HEAR-DS attribution.

```
@inproceedings{huwel2020hearing,
  title={Hearing aid research data set for acoustic environment recognition},
  author={H{\"u}wel, Andreas and Adilo{\u{g}}lu, Kamil and Bach, J{\"o}rg-Hendrik},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={706--710},
  year={2020},
  organization={IEEE}
}
```

CHiME 6 attribution.

```
@inproceedings{barker18_interspeech,
  author={Jon Barker and Shinji Watanabe and Emmanuel Vincent and Jan Trmal},
  title={{The Fifth 'CHiME' Speech Separation and Recognition Challenge: Dataset, Task and Baselines}},
  year=2018,
  booktitle={Proc. Interspeech 2018},
  pages={1561--1565},
  doi={10.21437/Interspeech.2018-1768}
}

@inproceedings{watanabe2020chime,
  title={CHiME-6 Challenge: Tackling multispeaker speech recognition for unsegmented recordings},
  author={Watanabe, Shinji and Mandel, Michael and Barker, Jon and Vincent, Emmanuel and Arora, Ashish and Chang, Xuankai and Khudanpur, Sanjeev and Manohar, Vimal and Povey, Daniel and Raj, Desh and others},
  booktitle={CHiME 2020-6th International Workshop on Speech Processing in Everyday Environments},
  year={2020}
}
```
