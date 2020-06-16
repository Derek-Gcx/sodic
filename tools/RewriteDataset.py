# 用于重写数据集条目为我们易于处理的格式
# 初步设想是按照每一天为一个文件,每一个entry为
# 编号       ---- [[路段, 速度, 时间], 
#           ----  [路段, 速度, 时间], 
#           ----  ......]
import time
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd
from tools.GetRoadID import getRoadID

GPS1 = "./asset/20191201_20191220.csv"
GPS2 = "./asset/201901_201903.csv"
GPS3 = "./asset/201910_11.csv"


def rewriteDataset():
    count = 0
    try:
        with open("./asset/20191201_20191220.csv", "r") as FP:
            line = FP.readline()
            days = set()
            entries = dict()

            while line:
                ok = 0

                for char in ["[", "]", "\""]:
                    line = line.replace(char, "")
                gps_records = line.split(",")[2:]

                # 读取一行,确定日期
                test = gps_records[0]   
                lng, lat, speed, direction, seconds = [eval(i) for i in test.strip().split(" ")]
                clock = list(time.gmtime(seconds))[:6]
                day = str(clock[1])+str(clock[2])
                if day not in days:
                    entries[day] = []
                    days.add(day)

                entry = []

                # 循环读取当前订单的每一条GPS记录,记录感兴趣路段的结果,加入到entry列表中
                for record in gps_records:

                    ok = (ok+1)%2
                    if ok == 0:
                        continue

                    lng, lat, speed, direction, seconds = [eval(i) for i in record.strip().split(" ")]
                    road_id = getRoadID(lng, lat, direction)

                    if road_id == 0:
                        continue
                    else:
                        clock = list(time.gmtime(seconds))[3:6]
                        entry.append([road_id, speed, clock])
                        
                
                if entry != []:
                    entries[day].append(entry)

                for d in entries.keys():
                    if len(entries[d]) == 1000:
                        with open(prefix+"2019"+str(d)+".csv", "a", newline="") as fp:
                            writer = csv.writer(fp)
                            writer.writerows(entries[d])
                        entries[d] = []
                        print("write into", prefix+"2019"+str(d)+".csv")

                count += 1
                line = FP.readline()
            
            for d in entries.keys():
                with open(prefix+"2019"+str(d)+".csv", "a", newline="") as fp:
                    writer = csv.writer(fp)
                    writer.writerows(entries[d])
                    entries[d] = []
                    print("write into", prefix+"2019"+str(d)+".csv")
    except :
        print("successfully processed {} lines".format(count))

def rewrite_dataset():
    dates = set()
    entries = dict()
    count = 0
    ok = 1
    try:
        with open(GPS1, "r") as fp:
            line = fp.readline()
            while line:
                count += 1
                # if count==1000:
                #     assert False
                for char in ["\"", "[", "]"]:
                    line = line.replace(char, "")
                gps_records = line.split(",")[2:]

                # 确定日期
                lng, lat, speed, direction, seconds = [eval(i) for i in gps_records[0].strip().split(" ")]
                time_stamp = time.gmtime(seconds+28800)
                date = str(time_stamp[1])+str(time_stamp[2])
                # if time_stamp[3] <=6 or time_stamp[3] >= 22:
                #     line = fp.readline()
                #     continue
                if date not in dates:
                    dates.add(date)
                    entries[date] = []

                df = [[eval(i) for i in j.strip().split(" ")] for j in gps_records]
                df = pd.DataFrame(data=df, columns=["lng", "lat", "speed", "direction", "seconds"])
                df["road_id"] = df.apply(lambda x:getRoadID(x["lng"], x["lat"], x["direction"]), axis=1)
                df["time"] = df.apply(lambda x:time.gmtime(x["seconds"]+28800)[3:6], axis=1)

                df = df[df["road_id"]!=0][["road_id", "speed", "time"]]

                if df.shape[0] <= 2:
                    line = fp.readline()
                    continue

                entries[date].append(df.values.tolist())

                for d in dates:
                    if len(entries[d]) == 1024:
                        with open("./train/201912/"+d+".csv", "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerows(entries[d])
                            entries[d] = []
                            print("wirte into", d, ".csv")

                line = fp.readline()
    except:
        print("successfully processed {} lines".format(count))
        for d in dates:
            print(len(entries[d]))






if __name__ == "__main__":
    # rewriteDataset()
    # showSpeed()
    rewrite_dataset()