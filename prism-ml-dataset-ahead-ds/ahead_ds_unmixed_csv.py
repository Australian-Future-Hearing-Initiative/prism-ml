"""
Produce CSV showing unmixed files belonging to 
train, validation and test set.
"""

import glob
import itertools
import pandas as pd
import sys


def _get_csv(
    dir_regex_01,
    dir_regex_02,
    dir_regex_03,
    dir_regex_04,
    dir_regex_05,
    dir_regex_06,
    dir_regex_07,
    dir_regex_08,
    dir_regex_09,
    dir_regex_10,
    dir_regex_11,
    dir_regex_12,
    dir_regex_13,
    dir_regex_14,
    dir_regex_15,
    dir_regex_16,
    dir_regex_17,
    dir_regex_18,
    dir_regex_19,
    dir_regex_20,
    datasplit,
    dbsplit,
    unmixed_csv,
):
    """Produce CSV showing unmixed files belonging to train, validation
    and test set.

    Args:
        dir_regex_01: Regex representing the directory of sounds
        for cocktail_party.
        dir_regex_02: Regex representing the directory of sounds
        for interfering_speakers.
        dir_regex_03: Regex representing the directory of sounds
        for in_traffic.
        dir_regex_04: Regex representing the directory of sounds
        for in_vehicle.
        dir_regex_05: Regex representing the directory of sounds
        for music.
        dir_regex_06: Regex representing the directory of sounds
        for quiet_indoors.
        dir_regex_07: Regex representing the directory of sounds
        for reverberant_environment.
        dir_regex_08: Regex representing the directory of sounds
        for wind_turbulence.
        dir_regex_09: Regex representing the directory of sounds
        for in_traffic_env.
        dir_regex_10: Regex representing the directory of sounds
        for in_vehicle_env.
        dir_regex_11: Regex representing the directory of sounds
        for music_env.
        dir_regex_12: Regex representing the directory of sounds
        for quiet_indoors_env.
        dir_regex_13: Regex representing the directory of sounds
        for reverberant_environment_env.
        dir_regex_14: Regex representing the directory of sounds
        for wind_turbulence_env.
        dir_regex_15: Regex representing the directory of sounds
        for in_traffic_speech.
        dir_regex_16: Regex representing the directory of sounds
        for in_vehicle_speech.
        dir_regex_17: Regex representing the directory of sounds
        for music_speech.
        dir_regex_18: Regex representing the directory of sounds
        for quiet_indoors_speech.
        dir_regex_19: Regex representing the directory of sounds
        for reverberant_environment_speech.
        dir_regex_20: Regex representing the directory of sounds
        for wind_turbulence_speech.
        all files, classes, and if they bolong to train/validation/test.
        datasplit: Split the data into train/validation/test
        according to this pattern. 0 - train, 1 - validation,
        2 - test.
        dbsplit: Integer to determine the order and SNR with which
        to mix sounds.
        unmixed_csv: Save CSV using this file name, containing list
        of unmixed files.
    """
    # Cocktail party and interfering speakers
    cocktail_party = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_01)
    ]
    interfering_speakers = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_02)
    ]
    # Environment sounds
    in_traffic = [x.split("/")[-1] for x in glob.glob(pathname=dir_regex_03)]
    in_vehicle = [x.split("/")[-1] for x in glob.glob(pathname=dir_regex_04)]
    music = [x.split("/")[-1] for x in glob.glob(pathname=dir_regex_05)]
    quiet_indoors = [x.split("/")[-1] for x in glob.glob(pathname=dir_regex_06)]
    reverberant_environment = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_07)
    ]
    wind_turbulence = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_08)
    ]
    # Environment sounds to be mixed with speech
    in_traffic_env = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_09)
    ]
    in_vehicle_env = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_10)
    ]
    music_env = [x.split("/")[-1] for x in glob.glob(pathname=dir_regex_11)]
    quiet_indoors_env = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_12)
    ]
    reverberant_environment_env = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_13)
    ]
    wind_turbulence_env = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_14)
    ]
    # Speech to be mixed with environment sounds
    in_traffic_speech = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_15)
    ]
    in_vehicle_speech = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_16)
    ]
    music_speech = [x.split("/")[-1] for x in glob.glob(pathname=dir_regex_17)]
    quiet_indoors_speech = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_18)
    ]
    reverberant_environment_speech = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_19)
    ]
    wind_turbulence_speech = [
        x.split("/")[-1] for x in glob.glob(pathname=dir_regex_20)
    ]
    # Combine list of file names
    file_lists = (
        cocktail_party
        + interfering_speakers
        + in_traffic
        + in_vehicle
        + music
        + quiet_indoors
        + reverberant_environment
        + wind_turbulence
        + in_traffic_env
        + in_vehicle_env
        + music_env
        + quiet_indoors_env
        + reverberant_environment_env
        + wind_turbulence_env
        + in_traffic_speech
        + in_vehicle_speech
        + music_speech
        + quiet_indoors_speech
        + reverberant_environment_speech
        + wind_turbulence_speech
    )

    # Associate each file with a class
    cocktail_party_list = ["cocktail_party"] * len(cocktail_party)
    interfering_speakers_list = ["interfering_speakers"] * len(
        interfering_speakers
    )
    in_traffic_list = ["in_traffic"] * len(in_traffic)
    in_vehicle_list = ["in_vehicle"] * len(in_vehicle)
    music_list = ["music"] * len(music)
    quiet_indoors_list = ["quiet_indoors"] * len(quiet_indoors)
    reverberant_environment_list = ["reverberant_environment"] * len(
        reverberant_environment
    )
    wind_turbulence_list = ["wind_turbulence"] * len(wind_turbulence)
    in_traffic_env_list = ["in_traffic_env"] * len(in_traffic_env)
    in_vehicle_env_list = ["in_vehicle_env"] * len(in_vehicle_env)
    music_env_list = ["music_env"] * len(music_env)
    quiet_indoors_env_list = ["quiet_indoors_env"] * len(quiet_indoors_env)
    reverberant_environment_env_list = ["reverberant_environment_env"] * len(
        reverberant_environment_env
    )
    wind_turbulence_env_list = ["wind_turbulence_env"] * len(
        wind_turbulence_env
    )
    in_traffic_speech_list = ["in_traffic_speech"] * len(in_traffic_speech)
    in_vehicle_speech_list = ["in_vehicle_speech"] * len(in_vehicle_speech)
    music_speech_list = ["music_speech"] * len(music_speech)
    quiet_indoors_speech_list = ["quiet_indoors_speech"] * len(
        quiet_indoors_speech
    )
    reverberant_environment_speech_list = [
        "reverberant_environment_speech"
    ] * len(reverberant_environment_speech)
    wind_turbulence_speech_list = ["wind_turbulence_speech"] * len(
        wind_turbulence_speech
    )
    # Combine all file associations
    file_association = (
        cocktail_party_list
        + interfering_speakers_list
        + in_traffic_list
        + in_vehicle_list
        + music_list
        + quiet_indoors_list
        + reverberant_environment_list
        + wind_turbulence_list
        + in_traffic_env_list
        + in_vehicle_env_list
        + music_env_list
        + quiet_indoors_env_list
        + reverberant_environment_env_list
        + wind_turbulence_env_list
        + in_traffic_speech_list
        + in_vehicle_speech_list
        + music_speech_list
        + quiet_indoors_speech_list
        + reverberant_environment_speech_list
        + wind_turbulence_speech_list
    )
    file_association_list_of_lists = [
        cocktail_party_list,
        interfering_speakers_list,
        in_traffic_list,
        in_vehicle_list,
        music_list,
        quiet_indoors_list,
        reverberant_environment_list,
        wind_turbulence_list,
        in_traffic_env_list,
        in_vehicle_env_list,
        music_env_list,
        quiet_indoors_env_list,
        reverberant_environment_env_list,
        wind_turbulence_env_list,
        in_traffic_speech_list,
        in_vehicle_speech_list,
        music_speech_list,
        quiet_indoors_speech_list,
        reverberant_environment_speech_list,
        wind_turbulence_speech_list,
    ]

    # Split data into train, validation, test
    def _data_split_helper():
        # 0 - train
        # 1 - validation
        # 2 - test
        repeating_seq = list()
        datasplit_list = list()
        for x in datasplit.split(","):
            if x.strip() == "0":
                datasplit_list.append("train")
            elif x.strip() == "1":
                datasplit_list.append("validation")
            elif x.strip() == "2":
                datasplit_list.append("test")
        for file_assoc_list in file_association_list_of_lists:
            single_repeating_seq = list(
                itertools.islice(
                    itertools.cycle(datasplit_list), len(file_assoc_list)
                )
            )
            repeating_seq.extend(single_repeating_seq)
        return repeating_seq

    data_split = _data_split_helper()

    # Split environment and speech data into the following SNR
    def _db_split_helper():
        repeating_seq = list()
        # The first part of the sequence contains only background not mixed with
        # Append NA to this first part
        for file_assoc_list in file_association_list_of_lists[0:8]:
            repeating_seq.extend(["NA"] * len(file_assoc_list))
        # Process the SNR
        dbsplit_list = [int(x) for x in dbsplit.split(",")]
        for file_assoc_list in file_association_list_of_lists[8:]:
            single_repeating_seq = list(
                itertools.islice(
                    itertools.cycle(dbsplit_list), len(file_assoc_list)
                )
            )
            repeating_seq.extend(single_repeating_seq)
        return repeating_seq

    db_split = _db_split_helper()

    # Create dataframe and save CSV
    unmixed_dict = {
        "file": file_lists,
        "file_association": file_association,
        "set": data_split,
        "SNR dB": db_split,
    }
    unmixed_pd = pd.DataFrame.from_dict(data=unmixed_dict)
    unmixed_pd.to_csv(
        path_or_buf=unmixed_csv, sep=",", header=True, index=False
    )


if __name__ == "__main__":
    dir_regex_01_arg = sys.argv[1]
    dir_regex_02_arg = sys.argv[2]
    dir_regex_03_arg = sys.argv[3]
    dir_regex_04_arg = sys.argv[4]
    dir_regex_05_arg = sys.argv[5]
    dir_regex_06_arg = sys.argv[6]
    dir_regex_07_arg = sys.argv[7]
    dir_regex_08_arg = sys.argv[8]
    dir_regex_09_arg = sys.argv[9]
    dir_regex_10_arg = sys.argv[10]
    dir_regex_11_arg = sys.argv[11]
    dir_regex_12_arg = sys.argv[12]
    dir_regex_13_arg = sys.argv[13]
    dir_regex_14_arg = sys.argv[14]
    dir_regex_15_arg = sys.argv[15]
    dir_regex_16_arg = sys.argv[16]
    dir_regex_17_arg = sys.argv[17]
    dir_regex_18_arg = sys.argv[18]
    dir_regex_19_arg = sys.argv[19]
    dir_regex_20_arg = sys.argv[20]
    datasplit_arg = sys.argv[21]
    dbsplit_arg = sys.argv[22]
    unmixed_csv_arg = sys.argv[23]

    _get_csv(
        dir_regex_01=dir_regex_01_arg,
        dir_regex_02=dir_regex_02_arg,
        dir_regex_03=dir_regex_03_arg,
        dir_regex_04=dir_regex_04_arg,
        dir_regex_05=dir_regex_05_arg,
        dir_regex_06=dir_regex_06_arg,
        dir_regex_07=dir_regex_07_arg,
        dir_regex_08=dir_regex_08_arg,
        dir_regex_09=dir_regex_09_arg,
        dir_regex_10=dir_regex_10_arg,
        dir_regex_11=dir_regex_11_arg,
        dir_regex_12=dir_regex_12_arg,
        dir_regex_13=dir_regex_13_arg,
        dir_regex_14=dir_regex_14_arg,
        dir_regex_15=dir_regex_15_arg,
        dir_regex_16=dir_regex_16_arg,
        dir_regex_17=dir_regex_17_arg,
        dir_regex_18=dir_regex_18_arg,
        dir_regex_19=dir_regex_19_arg,
        dir_regex_20=dir_regex_20_arg,
        datasplit=datasplit_arg,
        dbsplit=dbsplit_arg,
        unmixed_csv=unmixed_csv_arg,
    )
