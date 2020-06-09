import csv
import pandas as pd
import os
import sys
"""
    gcx: prepare dataset for LSTM
"""

sys.path.append(os.getcwd())

gp_num = 12
road_ids = [276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]

def devide_train_TTI_by_roadid():
    print("Start to devide train_TTI by road id.")

    entries = []
    df = pd.read_csv("./asset/train_TTI.csv")
    date = df["time"].apply(lambda x: x.split(" ")[0])
    clock = df["time"].apply(lambda x: x.split(" ")[1])
    df = df.drop("time", axis=1)
    df["date"] = date
    df["time"] = clock

    for road_id in set(df["id_road"]):
        save_csv = df[df["id_road"] == road_id]
        save_csv.to_csv("./train/processed/devided_train_TTI/"+str(road_id)+".csv", header=True, index=False)

    print("Finish deviding train_TTI by road id.")

def prepare_train_set_group12():
    print("Start to prepare training set.")
    devided_csv_path = "./train/processed/devided_train_TTI/"

    entries = []

    for road_id in road_ids:
        print("\tAdd {} training set.".format(road_id))
        df1 = pd.read_csv(devided_csv_path+str(road_id)+".csv")
        total = df1.shape[0]
        endpoint = 6
        while endpoint <= total-1:
            TTIs = list(df1.loc[endpoint-6:endpoint, "TTI"])
            entries.append(TTIs)
            endpoint += 1
        save_csv = pd.DataFrame(data=entries)
        save_csv.to_csv("./train/processed/to_train/train_"+str(road_id)+".csv", mode="w", header=False, index=False)
        entries = []

    print("Finish preparing training set.")



if __name__=="__main__":
    # devide_train_TTI_by_roadid()
    prepare_train_set_group12()