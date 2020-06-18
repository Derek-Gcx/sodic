import pandas as pd
import numpy as np
import csv
import os
import sys
import copy
import matplotlib.pyplot as plt


road_ids = [276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]
all_id = [276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]

def interested_date_range():
    month=12
    days = range(1, 30)
    clocks = list(range(8, 23))
    tot = pd.date_range(start="2019-01-01 12:00:00", end="2019-01-01 11:00:00", freq="10T")
    for day in days:
        tmp = pd.date_range(start="2019-12-%02d %02d:00:00"%(day, clocks[0]), end="2019-12-%02d %02d:00:00"%(day, clocks[-1]), freq="10T")
        tot = tot.append(tmp)
    return tot
def total_date_range():
    month=12
    days = range(1, 30)
    clocks = list(range(0, 24))
    tot = pd.date_range(start="2019-01-01 12:00:00", end="2019-01-01 11:00:00", freq="10T")
    for day in days:
        tmp = pd.date_range(start="2019-12-%02d %02d:00:00"%(day, clocks[0]), end="2019-12-%02d %02d:00:00"%(day, clocks[-1]), freq="10T")
        tot = tot.append(tmp)
    return tot

def merge_dataset():
    for road_id in all_id:
        df1 = pd.read_csv("./test/processed/devided_train_TTI/"+str(road_id)+".csv", index_col=0)
        df2 = pd.read_csv("./test/processed/feature/test/"+str(road_id)+".csv", index_col=0, header=None)
        df2.columns=["avg", "low2"]

        df1 = df1.set_index(pd.to_datetime(df1.index))
        df2 = df2.set_index(pd.to_datetime(df2.index))
        # 添加时区转换
        # df2 = df2.set_index(pd.to_datetime(pd.to_datetime(df2.index, utc=True).tz_convert("Asia/SHanghai").strftime("%Y-%m-%d %H:%M:%S")))

        interested = interested_date_range()
        tot = total_date_range()
        df1 = df1.reindex(index=tot, fill_value=0)
        del df1["id_road"]
        df2 = df2.reindex(index=tot, fill_value=0)
        # df2["margin"] = df2["max"]-df2["min"]
        # del df2["max"]
        # del df2["min"]

        df = df1.join(df2)
        speed1 = df["avg"].values
        speed2 = copy.deepcopy(speed1)[1:]
        speed3 = copy.deepcopy(speed1)[2:]
        speed3 = 0.6*speed3+0.3*speed2[:-1]+0.1*speed1[:-2]
        speed3 = speed3.tolist()
        speed3.insert(0, 0)
        speed3.insert(0, 0)
        df["avg"] = speed3

        a = df["TTI"].iloc[1:].values
        df = df.iloc[:-1]
        df["label"] = a
        df["label"] = df["label"] * 5
        

        df["speed"] = df["speed"]/3.6
        # df["low1"] = df["low1"] * 5
        df["low2"] = df["low2"] * 5
        # df["high1"] = df["high1"] * 5
        # df["high2"] = df["high1"] * 5
        # df["margin"] = df["high1"]-df["low1"]
        # df["TTI"] = df["TTI"]*10
        # df["count"] = df["count"] / 10

        df = df.reindex(index=interested, fill_value=0)

        df["TTI"] = df["TTI"]*10
        # df["label"] = df["label"] * 10

        print(df.corr())

        df.iloc[:80].plot()
        plt.show()


        # fig = copy.deepcopy(df)
        # fig["TTI"] = fig["TTI"]*10
        # fig["speed"] = fig["speed"]/3.6
        # a = fig["avg"].values
        # b = copy.deepcopy(a)[1:]
        # c = copy.deepcopy(a)[2:]
        # c = 0.6*c+0.3*b[:-1]+0.1*a[:-2]
        # # b /=2
        # # b = b.tolist()
        # # b.insert(0, 0)
        # c = c.tolist()
        # c.insert(0, 0)
        # c.insert(0,0)
        # fig["avg"] = c
        
        # fig["margin"] = fig["max"]-fig["min"]
        # del fig["count"]
        # fig.iloc[0:80].plot()
        # plt.title(str(road_id))
        # plt.show()
        # df.to_csv("./train/processed/to_train/"+str(road_id)+".csv")





if __name__ == "__main__":
    merge_dataset()