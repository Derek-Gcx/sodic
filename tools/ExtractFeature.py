import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import os

road_ids = [276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]
user_road_time = None

GPS1 = "./train/201912/"
# GPS2 = "./train/201910_11/"
# GPS3 = "./train/201901_201903/"
GPSs = [
    GPS1, 
    # GPS2, 
    # GPS3, 
]


def extract_feature():
    out_path = "./train/processed/feature/201912/"
    out_csvs = {}

    for GPS in GPSs:
        for file in list(os.walk(GPS))[0][2]:
            print("processing", GPS+file)
            with open(GPS+file, "r") as fp:
                entries = fp.readlines()
            entries = [[eval(j) for j in i[1:-2].split("\",\"")] for i in entries]
            entries = pd.DataFrame(entries)

            global user_road_time
            user_road_time = pd.DataFrame(columns=["road_id", "time", "avg", "low_speed_ratio", "high_speed_ratio", "is_low", "is_high"])
            entries.apply(lambda x: merge(get_feature(x, file[0:4])), axis=1)

            prod = user_road_time.groupby(by=["road_id", "time"], as_index=False).agg({"avg": "mean", "low_speed_ratio": "mean", "high_speed_ratio": "mean", "is_low": "mean", "is_high": "mean"})

            for road_id in prod["road_id"].unique():
                out_csvs[road_id] = prod[prod["road_id"]==road_id].iloc[:, 1:]
                out_csvs[road_id].to_csv(out_path+str(road_id)+".csv", mode="a", header=False, index=False)


def merge(b):
    global user_road_time
    user_road_time = user_road_time.append(b)

def get_feature(x: pd.Series, date):
    x = x.dropna()
    name = x.name
    x = pd.DataFrame(x.to_list(), columns=["road_id", "speed", "clock"])
    x = x[x["speed"] <= 30]
    x["clock"] = x.apply(lambda x: pd.Timestamp(2019, int(date[0:2]), int(date[2:]), x["clock"][0], x["clock"][1]//10*10, 0), axis=1)

    ret = x.groupby(by=["road_id", "clock"], as_index=False).agg({"speed": ["mean", lambda x:len(x), lambda x:sum(x<=6)/len(x), lambda x:sum(x>=13)/len(x)]})

    ret.columns=["road_id", "time", "avg", "count", "low_speed_ratio", "high_speed_ratio"]
    ret = ret[ret["count"]>=5]

    ret["is_low"] = (ret["avg"] <= 6).astype(np.float64)
    ret["is_high"] = (ret["avg"] >= 14).astype(np.float64)


    del ret["count"]
    return ret

if __name__ == "__main__":
    extract_feature()