---
language:
- en
license: cc-by-sa-4.0
tags:
- audio
task_categories:
- audio-classification
---

# Another HEaring AiD DataSet (AHEAD-DS) unmixed
Another HEaring AiD DataSet (AHEAD-DS) unmixed is an audio dataset labelled with audiologically relevant scene categories for hearing aids. This dataset contains the environment and speech sounds before they were mixed. The file ahead_ds_unmixed.csv documents the details of every file.

* [Website](https://github.com/Australian-Future-Hearing-Initiative)
* [Paper](https://arxiv.org/abs/2508.10360)
* [Code](https://github.com/Australian-Future-Hearing-Initiative/prism-ml/prism-ml-yamnetp-tune)
* [Dataset AHEAD-DS](https://huggingface.co/datasets/hzhongresearch/ahead_ds)
* [Dataset AHEAD-DS unmixed](https://huggingface.co/datasets/hzhongresearch/ahead_ds_unmixed)
* [Models](https://huggingface.co/hzhongresearch/yamnetp_ahead_ds)

## Description of data
All files are encoded as single channel WAV, 16 bit signed, sampled at 16 kHz with 10 seconds per recording.

| file_association               | Description                                  |
|:-------------------------------|:---------------------------------------------|
| cocktail_party                 | cocktail_party sounds                        |
| interfering_speakers           | interfering_speakers sounds                  |
| in_traffic                     | in_traffic sounds                            |
| in_vehicle                     | in_vehicle sounds                            |
| music                          | music sounds                                 |
| quiet_indoors                  | quiet_indoors sounds                         |
| reverberant_environment        | reverberant_environment sounds               |
| wind_turbulence                | wind_turbulence sounds                       |
| in_traffic_env                 | speech_in_traffic environment sounds         |
| in_vehicle_env                 | speech_in_vehicle environment sounds         |
| music_env                      | speech_in_music environment sounds           |
| quiet_indoors_env              | speech_in_quiet_indoors environment sounds   |
| reverberant_environment_env    | speech_in_reverberant_environment sounds     |
| wind_turbulence_env            | speech_in_wind_turbulence environment sounds |
| in_traffic_speech              | speech_in_traffic speech                     |
| in_vehicle_speech              | speech_in_vehicle speech                     |
| music_speech                   | speech_in_music speech                       |
| quiet_indoors_speech           | speech_in_quiet_indoors speech               |
| reverberant_environment_speech | speech_in_reverberant_environment speech     |
| wind_turbulence_speech         | speech_in_wind_turbulence speech             |

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
