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
    out_path = "./train/processed/feature/"
    out_csvs = {}
    # for road_id in road_ids:
        # out_csvs[road_id] = pd.DataFrame()

    for GPS in GPSs:
        for file in list(os.walk(GPS))[0][2]:
            print("processing", GPS+file)
            with open(GPS+file, "r") as fp:
                entries = fp.readlines()
            entries = [[eval(j) for j in i[1:-2].split("\",\"")] for i in entries]
            entries = pd.DataFrame(entries)

            global user_road_time
            user_road_time = pd.DataFrame(columns=["id", "road_id", "time", "speed_avg_car", "speed_min_car", "speed_max_car", "speed_var_car"])
            entries["speed_avg_car"] = entries.apply(lambda x: merge(get_feature(x, file[0:4])), axis=1)
            # entries["speed_var_car"] = entries.apply(get_speed_var_car, axis=1)
            # entries["speed_range"] = entries.apply(get_speed_range_car, axis=1)

            prod = user_road_time.groupby(by=["road_id", "time"]).agg({"id": lambda x:len(x), "speed_avg_car": "mean", "speed_min_car": "min", "speed_max_car": "max", "speed_var_car": "mean"})
            print(prod.columns)
            for road_id in prod["road_id"].unique():
                out_csvs[road_id] = prod[prod["road_id"]==road_id][1:]
                out_csvs[road_id].to_csv(out_path+str(road_id)+".csv", mode="a", header=False)
                # out_csvs[road_id] = pd.DataFrame()
            assert False




def merge(b):
    global user_road_time
    user_road_time = user_road_time.append(b)

def get_feature(x: pd.Series, date):
    x = x.dropna()
    name = x.name
    x = pd.DataFrame(x.to_list(), columns=["road_id", "speed", "clock"])
    x["id"] = name
    x["clock"] = x.apply(lambda x: pd.Timestamp(2019, int(date[0:2]), int(date[2:]), x["clock"][0], x["clock"][1]//10*10, 0), axis=1)
    ret = x.groupby(by=["id", "road_id", "clock"], as_index=False).agg({"speed": ["mean", "min", "max", lambda x:x.var()]})
    ret.columns=["id", "road_id", "time", "speed_avg_car", "speed_min_car", "speed_max_car", "speed_var_car"]

    return ret
    
    

# def get_speed_var_car(x):

# def get_speed_range_car(x):


            


if __name__ == "__main__":
    extract_feature()