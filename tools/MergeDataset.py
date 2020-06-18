import pandas as pd
import numpy as np
import csv
import sys
import os
import copy
import matplotlib.pyplot as plt
import datetime

"""
merge train_TTI and feature csvs like train_TTI
and
generate training set and testing set
"""

all_id = [276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]

train_feature_csvs = [
    "./train/processed/feature/201901_201903/", 
    "./train/processed/feature/201910_11/", 
    "./train/processed/feature/201912/", 
]
test_feature_csvs = "./test/processed/feature/test/"

devided_train_TTI = "./train/processed/devided_train_TTI/"
devided_test_TTI = "./test/processed/devided_train_TTI/"

train_out_path = "./train/processed/merge/"
test_out_path = "./test/processed/merge/"


def devide_train_TTI_by_roadid():
    # 将train_TTI分割成12个数据集
    print("Start to devide train_TTI by road id.")

    entries = []
    df = pd.read_csv("./asset/train_TTI.csv")

    for road_id in set(df["id_road"]):
        save_csv = df[df["id_road"] == road_id]
        save_csv.to_csv(devided_train_TTI+str(road_id)+".csv", header=True, index=False)

    # 填充缺失值
    print("Start to check and fill nan.")
    date_range1 = pd.date_range(start="2019-01-01 00:00:00", end="2019-03-31 23:50:00", freq="10T")
    date_range2 = pd.date_range(start="2019-10-01 00:00:00", end="2019-12-21 23:50:00", freq="10T")
    
    for road_id in all_id:
        df = pd.read_csv("./train/processed/devided_train_TTI/"+str(road_id)+".csv")
        df = df.set_index("time")
        df = df.set_index(pd.to_datetime(df.index))

        new_df1 = df.reindex(date_range1, fill_value=None)
        new_df2 = df.reindex(date_range2, fill_value=None)
        new_df = new_df1.append(new_df2)

        new_df.to_csv("./train/processed/devided_train_TTI/"+str(road_id)+".csv", header=True, index=True)
    print("Finish filling nan.")
    print("Finish deviding train_TTI by road id.")

def devide_test_TTI_by_roadid():
    print("Start to devide test TTI by road id.")
    testTTI = pd.read_csv("./test/toPredict_train_TTI.csv", index_col=None)
    for road_id in all_id:
        save_csv = testTTI[testTTI["id_road"] == road_id]
        save_csv = save_csv.set_index("time")
        save_csv = save_csv.set_index(pd.to_datetime(save_csv.index))
        save_csv.to_csv("./test/processed/devided_train_TTI/"+str(road_id)+".csv", header=True, index=True)
    print("Finish deviding test TTI by road id.")

    print("Filling nan for test.")
    date1 = ["2019-12-21", "2019-12-22", "2019-12-23", "2019-12-24", "2019-12-25", "2019-12-26"]
    clock1 = ["07:30:00", "09:30:00", "11:30:00", "13:30:00", "15:30:00", "17:30:00", "19:30:00"]
    date2 = ["2019-12-27", "2019-12-28", "2019-12-29", "2019-12-30", "2019-12-31", "2020-01-01"]
    clock2 = ["08:00:00", "10:00:00", "12:00:00", "14:00:00", "16:00:00", "18:00:00", "20:00:00"]
    date_ranges = []
    for date in date1:
        for clock in clock1:
            tmp = pd.date_range(start=" ".join([date, clock]), periods=9, freq="10T")
            date_ranges.append(tmp)
    for date in date2:
        for clock in clock2:
            tmp = pd.date_range(start=" ".join([date, clock]), periods=9, freq="10T")
            date_ranges.append(tmp)
    for road_id in all_id:
        df = pd.read_csv("./test/processed/devided_train_TTI/"+str(road_id)+".csv", index_col=0)
        df = df.set_index(pd.to_datetime(df.index))
        new_dfs = [df.reindex(i, fill_value=0) for i in date_ranges]
        new_df = pd.concat(new_dfs, axis=0)

        new_df.to_csv("./test/processed/devided_train_TTI/"+str(road_id)+".csv", header=True, index=True)



def merge_feature_into_test_TTI():
    print("Merging feature into test TTI.")
    for road_id in all_id:
        testTTI = pd.read_csv(devided_test_TTI + str(road_id) + ".csv", index_col=0)
        testTTI = testTTI.set_index(pd.to_datetime(testTTI.index))

        feature = pd.read_csv(test_feature_csvs + str(road_id) + ".csv", header=None, index_col=None)
        feature.columns = ["time", "avg", "low_ratio"]
        # 将特征集中的2019-01-01改成2020-01-01
        feature["time"] = feature["time"].apply(lambda x: "2020-01-01"+x[10:] if x.startswith("2019-01-01") else x)
        feature = feature.set_index("time")
        feature = feature.set_index(pd.to_datetime(feature.index))
        
        merged = testTTI.join(feature)

        # 如果需要加什么其他的特征，在这里加
        merged["time_block"] = [x.hour*6 + x.minute//10 for x in merged.index]
        merged["day_of_week"] = [x.dayofweek for x in merged.index]
        merged["is_weekend"] = merged["day_of_week"].apply(lambda x: 1 if x==6 or x==5 else 0)

        # 缩放相关
        merged["speed"] = merged["speed"] / 3.6

        # 保存到merge目录下
        merged.to_csv(test_out_path+str(road_id)+".csv", header=True, index=True)

def merge_feature_into_train_TTI():
    print("Merging feature into train TTI.csv")
    for road_id in all_id:
        trainTTI = pd.read_csv(devided_train_TTI + str(road_id) + ".csv", index_col=0)
        trainTTI = trainTTI.set_index(pd.to_datetime(trainTTI.index))

        tmp = pd.DataFrame(columns=["avg", "low_ratio"])
        for train_feature in train_feature_csvs:
            feature = pd.read_csv(train_feature + str(road_id) + ".csv", header=None, index_col=0)
            feature.columns = ["avg", "low_ratio"]
            feature = feature.set_index(pd.to_datetime(feature.index))

            tmp = tmp.append(feature)
        
        merged = trainTTI.join(tmp)
        merged = merged.iloc[:-144]

        # 如果需要加什么特征，在这里加
        merged["time_block"] = [x.hour*6 + x.minute//10 for x in merged.index]
        merged["day_of_week"] = [x.dayofweek for x in merged.index]
        merged["is_weekend"] = merged["day_of_week"].apply(lambda x: 1 if x==6 or x==5 else 0)


        # 缩放相关
        merged["speed"] = merged["speed"] / 3.6

        merged.to_csv(train_out_path+str(road_id)+".csv", header=True, index=True)

def check_nan_values():
    # train
    for road_id in all_id:
        df = pd.read_csv(train_out_path + str(road_id) + ".csv")
        pass



if __name__ == "__main__":
    # devide_train_TTI_by_roadid()
    devide_test_TTI_by_roadid()
    merge_feature_into_test_TTI()
    # merge_feature_into_train_TTI()
    # check_nan_values()