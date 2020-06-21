import os
import pandas as pd
import numpy as np


def trainset():
    """
    create a time series training dataset, with 6*7 dims feaures and 3 dims target variable
    """
    path = "./train/train_features"
    out_path = "./train/processed/train_feature/"
    for name in list(os.listdir(path)):
        with open(path + "/" + name) as f:
            data = f.readlines()
        col_names = data[0].replace("\n", "").split(",")
        col_names[0] = "day"
        data = [(s.replace("\n", "")).split(",") for s in data[1:]]
        df = pd.DataFrame(data)
        df.columns = col_names
        df[col_names[0]] = df[col_names[0]].apply(lambda x: x.split(" ")[0])
        df[col_names[0]] = df[col_names[0]].apply(
            lambda x: 10000 * int(x.split("-")[0]) + 100 * int(
                x.split("-")[1]) + int(x.split("-")[2]))
        for col in col_names[1:]:
            df[col] = pd.to_numeric(df[col])
        # the time slice between 7:00-22:00
        df.drop(df[df["time_block"] < 45].index, inplace=True)
        df.drop(df[df["time_block"] > 135].index, inplace=True)
        days = list(df["day"])
        out = []
        days = sorted(list(set(days)))
        for day in days:
            temp_df = df[df.day == day]
            for _, row in temp_df.iterrows():
                block = row["time_block"]
                result = temp_df[(temp_df.time_block >= (block - 8))
                                 & (temp_df.time_block <= block)]
                if (len(result) == 9):
                    res = np.array(result.values)[:6, 2:].ravel()
                    res = np.append(res, result.values[6:, 2])
                    out.append(list(res))
        out = pd.DataFrame(out).dropna()
        out.to_csv(out_path + name, mode="w+", header=False, index=False)


def testset():
    """
    create a test set similar to trian set with the last 3 dims filled by zero
    """
    path = "./train/test_features"
    out_path = "./train/processed/test_feature/"
    data_list = []
    # store the data by the order in toPredict_noLabel
    for name in [
            "276183", "276184", "275911", "275912", "276240", "276241",
            "276264", "276265", "276268", "276269", "276737", "276738"]:
        with open(path + "/" + name + ".csv") as f:
            data = f.readlines()
        col_names = data[0].replace("\n", "").split(",")
        col_names[0] = "day"
        data = [(s.replace("\n", "")).split(",") for s in data[1:]]
        df = pd.DataFrame(data)
        df.columns = col_names
        df[col_names[0]] = df[col_names[0]].apply(lambda x: x.split(" ")[0])
        df[col_names[0]] = df[col_names[0]].apply(
            lambda x: 10000 * int(x.split("-")[0]) + 100 * int(x.split("-")[1]) + int(x.split("-")[2]))
        for col in col_names[1:]:
            df[col] = pd.to_numeric(df[col])
        data_list.append(df)
    days = list(data_list[0]["day"])
    out = []
    # write data by the order of date
    days = sorted(list(set(days)))
    for day in days:
        for df in data_list:
            temp_df = df[df.day == day]
            for _, row in temp_df.iterrows():
                block = row["time_block"]
                result = temp_df[(temp_df.time_block >= (block - 8))
                                 & (temp_df.time_block <= block)]
                if (len(result) == 9):
                    res = np.array(result.values)[:6, 2:].ravel()
                    res = np.append(res, result.values[6:, 2])
                    out.append(list(res))
    out = pd.DataFrame(out).dropna()
    out.to_csv(out_path + "ToPredict.csv",
               mode="w+",
               header=False,
               index=False)


if __name__ == '__main__':
    trainset()
    testset()