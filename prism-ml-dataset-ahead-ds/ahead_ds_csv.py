"""
Produce CSV showing files belonging to train, 
validation and test set.
"""

import glob
import itertools
import pandas as pd
import sys


def _get_csv(
    class1_regex,
    class2_regex,
    class3_regex,
    class4_regex,
    class5_regex,
    class6_regex,
    class7_regex,
    class8_regex,
    class9_regex,
    class10_regex,
    class11_regex,
    class12_regex,
    class13_regex,
    class14_regex,
    class1_name,
    class2_name,
    class3_name,
    class4_name,
    class5_name,
    class6_name,
    class7_name,
    class8_name,
    class9_name,
    class10_name,
    class11_name,
    class12_name,
    class13_name,
    class14_name,
    all_csv,
    train_csv,
    val_csv,
    test_csv,
    datasplit,
):
    """Produce CSV showing files belonging to train, validation
    and test set.

    Args:
        class1_regex: Regex representing the directory of class 1.
        class2_regex: Regex representing the directory of class 2.
        class3_regex: Regex representing the directory of class 3.
        class4_regex: Regex representing the directory of class 4.
        class5_regex: Regex representing the directory of class 5.
        class6_regex: Regex representing the directory of class 6.
        class7_regex: Regex representing the directory of class 7.
        class8_regex: Regex representing the directory of class 8.
        class9_regex: Regex representing the directory of class 9.
        class10_regex: Regex representing the directory of class 10.
        class11_regex: Regex representing the directory of class 11.
        class12_regex: Regex representing the directory of class 12.
        class13_regex: Regex representing the directory of class 13.
        class14_regex: Regex representing the directory of class 14.
        class1_name: Name of class 1.
        class2_name: Name of class 2.
        class3_name: Name of class 3.
        class4_name: Name of class 4.
        class5_name: Name of class 5.
        class6_name: Name of class 6.
        class7_name: Name of class 7.
        class8_name: Name of class 8.
        class9_name: Name of class 9.
        class10_name: Name of class 10.
        class11_name: Name of class 11.
        class12_name: Name of class 12.
        class13_name: Name of class 13.
        class14_name: Name of class 14.
        all_csv: Save CSV using this file name, contains files and classes.
        train_csv: Save CSV using this file name, files in training set.
        val_csv: Save CSV using this file name, files in validation set.
        test_csv: Save CSV using this file name, files in test set.
        datasplit: Split the data into train/validation/test
        according to this pattern. 0 - train, 1 - validation,
        2 - test.
    """
    # Class files
    class1_files = [x.split("/")[-1] for x in glob.glob(pathname=class1_regex)]
    class2_files = [x.split("/")[-1] for x in glob.glob(pathname=class2_regex)]
    class3_files = [x.split("/")[-1] for x in glob.glob(pathname=class3_regex)]
    class4_files = [x.split("/")[-1] for x in glob.glob(pathname=class4_regex)]
    class5_files = [x.split("/")[-1] for x in glob.glob(pathname=class5_regex)]
    class6_files = [x.split("/")[-1] for x in glob.glob(pathname=class6_regex)]
    class7_files = [x.split("/")[-1] for x in glob.glob(pathname=class7_regex)]
    class8_files = [x.split("/")[-1] for x in glob.glob(pathname=class8_regex)]
    class9_files = [x.split("/")[-1] for x in glob.glob(pathname=class9_regex)]
    class10_files = [
        x.split("/")[-1] for x in glob.glob(pathname=class10_regex)
    ]
    class11_files = [
        x.split("/")[-1] for x in glob.glob(pathname=class11_regex)
    ]
    class12_files = [
        x.split("/")[-1] for x in glob.glob(pathname=class12_regex)
    ]
    class13_files = [
        x.split("/")[-1] for x in glob.glob(pathname=class13_regex)
    ]
    class14_files = [
        x.split("/")[-1] for x in glob.glob(pathname=class14_regex)
    ]
    all_files = (
        class1_files
        + class2_files
        + class3_files
        + class4_files
        + class5_files
        + class6_files
        + class7_files
        + class8_files
        + class9_files
        + class10_files
        + class11_files
        + class12_files
        + class13_files
        + class14_files
    )
    # Class length
    class1_l = len(class1_files)
    class2_l = len(class2_files)
    class3_l = len(class3_files)
    class4_l = len(class4_files)
    class5_l = len(class5_files)
    class6_l = len(class6_files)
    class7_l = len(class7_files)
    class8_l = len(class8_files)
    class9_l = len(class9_files)
    class10_l = len(class10_files)
    class11_l = len(class11_files)
    class12_l = len(class12_files)
    class13_l = len(class13_files)
    class14_l = len(class14_files)

    # Class onehot vectors
    def _class_v(v_index):
        v_seq = [
            class1_l,
            class2_l,
            class3_l,
            class4_l,
            class5_l,
            class6_l,
            class7_l,
            class8_l,
            class9_l,
            class10_l,
            class11_l,
            class12_l,
            class13_l,
            class14_l,
        ]
        class_v = list()
        for index, lengths in enumerate(v_seq):
            # Assign the positions specified
            # by v_index equal to 1
            if (index + 1) == v_index:
                class_v.extend([1] * lengths)
            else:
                class_v.extend([0] * lengths)
        return class_v

    class1_v = _class_v(1)
    class2_v = _class_v(2)
    class3_v = _class_v(3)
    class4_v = _class_v(4)
    class5_v = _class_v(5)
    class6_v = _class_v(6)
    class7_v = _class_v(7)
    class8_v = _class_v(8)
    class9_v = _class_v(9)
    class10_v = _class_v(10)
    class11_v = _class_v(11)
    class12_v = _class_v(12)
    class13_v = _class_v(13)
    class14_v = _class_v(14)

    # Split data into train, validation, test
    def _data_split():
        # 0 - training
        # 1 - validation
        # 2 - testing
        datasplit_part = [int(x) for x in datasplit.split(",")]
        split_seq = [
            class1_l,
            class2_l,
            class3_l,
            class4_l,
            class5_l,
            class6_l,
            class7_l,
            class8_l,
            class9_l,
            class10_l,
            class11_l,
            class12_l,
            class13_l,
            class14_l,
        ]
        split_v = list()
        for lengths in split_seq:
            class_split = list(
                itertools.islice(itertools.cycle(datasplit_part), lengths)
            )
            split_v.extend(class_split)
        return split_v

    data_split = _data_split()
    data_split_pd = pd.DataFrame.from_dict(data={"set": data_split})
    # Create the dataset CSVs
    ahead_dict = {
        "files": all_files,
        class1_name: class1_v,
        class2_name: class2_v,
        class3_name: class3_v,
        class4_name: class4_v,
        class5_name: class5_v,
        class6_name: class6_v,
        class7_name: class7_v,
        class8_name: class8_v,
        class9_name: class9_v,
        class10_name: class10_v,
        class11_name: class11_v,
        class12_name: class12_v,
        class13_name: class13_v,
        class14_name: class14_v,
    }
    ahead_df = pd.DataFrame.from_dict(data=ahead_dict)
    # Save whole dataset
    ahead_df.to_csv(path_or_buf=all_csv, sep=",", header=True, index=False)
    # Save training data
    ahead_df_train = ahead_df[data_split_pd["set"] == 0]
    ahead_df_train.to_csv(
        path_or_buf=train_csv, sep=",", header=True, index=False
    )
    # Save validation data
    ahead_df_val = ahead_df[data_split_pd["set"] == 1]
    ahead_df_val.to_csv(path_or_buf=val_csv, sep=",", header=True, index=False)
    # Save testing data
    ahead_df_test = ahead_df[data_split_pd["set"] == 2]
    ahead_df_test.to_csv(
        path_or_buf=test_csv, sep=",", header=True, index=False
    )


if __name__ == "__main__":
    class1_regex_arg = sys.argv[1]
    class2_regex_arg = sys.argv[2]
    class3_regex_arg = sys.argv[3]
    class4_regex_arg = sys.argv[4]
    class5_regex_arg = sys.argv[5]
    class6_regex_arg = sys.argv[6]
    class7_regex_arg = sys.argv[7]
    class8_regex_arg = sys.argv[8]
    class9_regex_arg = sys.argv[9]
    class10_regex_arg = sys.argv[10]
    class11_regex_arg = sys.argv[11]
    class12_regex_arg = sys.argv[12]
    class13_regex_arg = sys.argv[13]
    class14_regex_arg = sys.argv[14]
    class1_name_arg = sys.argv[15]
    class2_name_arg = sys.argv[16]
    class3_name_arg = sys.argv[17]
    class4_name_arg = sys.argv[18]
    class5_name_arg = sys.argv[19]
    class6_name_arg = sys.argv[20]
    class7_name_arg = sys.argv[21]
    class8_name_arg = sys.argv[22]
    class9_name_arg = sys.argv[23]
    class10_name_arg = sys.argv[24]
    class11_name_arg = sys.argv[25]
    class12_name_arg = sys.argv[26]
    class13_name_arg = sys.argv[27]
    class14_name_arg = sys.argv[28]
    all_csv_arg = sys.argv[29]
    train_csv_arg = sys.argv[30]
    val_csv_arg = sys.argv[31]
    test_csv_arg = sys.argv[32]
    datasplit_arg = sys.argv[33]

    _get_csv(
        class1_regex=class1_regex_arg,
        class2_regex=class2_regex_arg,
        class3_regex=class3_regex_arg,
        class4_regex=class4_regex_arg,
        class5_regex=class5_regex_arg,
        class6_regex=class6_regex_arg,
        class7_regex=class7_regex_arg,
        class8_regex=class8_regex_arg,
        class9_regex=class9_regex_arg,
        class10_regex=class10_regex_arg,
        class11_regex=class11_regex_arg,
        class12_regex=class12_regex_arg,
        class13_regex=class13_regex_arg,
        class14_regex=class14_regex_arg,
        class1_name=class1_name_arg,
        class2_name=class2_name_arg,
        class3_name=class3_name_arg,
        class4_name=class4_name_arg,
        class5_name=class5_name_arg,
        class6_name=class6_name_arg,
        class7_name=class7_name_arg,
        class8_name=class8_name_arg,
        class9_name=class9_name_arg,
        class10_name=class10_name_arg,
        class11_name=class11_name_arg,
        class12_name=class12_name_arg,
        class13_name=class13_name_arg,
        class14_name=class14_name_arg,
        all_csv=all_csv_arg,
        train_csv=train_csv_arg,
        val_csv=val_csv_arg,
        test_csv=test_csv_arg,
        datasplit=datasplit_arg,
    )
