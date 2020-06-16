# 用于重写数据集条目为我们易于处理的格式
# 初步设想是按照每一天为一个文件,每一个entry为
# 编号       ---- [[路段, 速度, 时间], 
#           ----  [路段, 速度, 时间], 
#           ----  ......]
import time
import csv
import matplotlib.pyplot as plt
import math
import sys
import os
import numpy as np
from collections import defaultdict

prefix = "./train/processed/"
sys.path.append(os.getcwd())
from tools.GetRoadID import getRoadID
import tools.group as gp

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


def calculate(buffer):
    buffer = np.array(buffer)
    start = time.gmtime(buffer[0][1])
    length = len(buffer)
    result = []
    index = 0
    next_ten = (9 - (start.tm_min % 10)) * 20 + int((60 - start.tm_sec)/3) + 1
    if length > next_ten:
        if start.tm_hour > 10:
            result.append(["{}-{}-{} {}:{}0:00".format(start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, int(start.tm_min/10)), np.mean(buffer[0:next_ten, 0])])
        else:
            result.append(["{}-{}-{} 0{}:{}0:00".format(start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, int(start.tm_min/10)), np.mean(buffer[0:next_ten, 0])])
        length -= next_ten
        index += next_ten
        start = time.gmtime(buffer[index][1])
        next_ten = (9 - (start.tm_min % 10)) * 20 + int((60 - start.tm_sec)/3) + 1
        while(length - next_ten > 0):
            if start.tm_hour > 9:
                result.append(["{}-{}-{} {}:{}0:00".format(start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, int(start.tm_min/10)), np.mean(buffer[index:index+next_ten, 0])])
            else:
                result.append(["{}-{}-{} 0{}:{}0:00".format(start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, int(start.tm_min/10)), np.mean(buffer[index:index+next_ten, 0])])
            length -= next_ten
            index += next_ten
            if index != len(buffer):
                start = time.gmtime(buffer[index][1])
                next_ten = (9 - (start.tm_min % 10)) * 20 + int((60 - start.tm_sec)/3) + 1
    if(length != 0):
        if start.tm_hour > 9:
            result.append(["{}-{}-{} {}:{}0:00".format(start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, int(start.tm_min/10)), np.mean(buffer[index:, 0])])
        else:
            result.append(["{}-{}-{} 0{}:{}0:00".format(start.tm_year, start.tm_mon, start.tm_mday, start.tm_hour, int(start.tm_min/10)), np.mean(buffer[index:, 0])])
    return result


def extractData():
    dict_list = []
    for i in range(12):
        dict_list.append(defaultdict(list))
    for line in open("./asset/gps/20191201_20191220.csv", "r"):
        for char in ["[", "]", "\""]:
            line = line.replace(char, "")
        line = line.split(",")[2:]
        road_id = 0
        buffer = []
        for record in line:
            lng, lat, speed, direction, seconds = [eval(i) for i in record.strip().split(" ")]
            temp = getRoadID(lng, lat, direction)
            if temp != 0:
                if temp == road_id:
                    buffer.append([speed, seconds])
                else:
                    if buffer != []:
                        processed = calculate(buffer)
                        for entry in processed:
                            dict_list[gp.igmap(road_id)][entry[0]].append(entry[1])
                    road_id = temp
                    buffer = []
                    buffer.append([speed, seconds])
            if buffer != []:
                processed = calculate(buffer)
                for entry in processed:
                    dict_list[gp.igmap(road_id)][entry[0]].append(entry[1])
    for i in range(12):
        data_dict = dict_list[i]
        with open("./train/processed/gps/" + str(i) + ".csv", "w+", newline='') as objfile:
            obj_writer = csv.writer(objfile)
            for key in data_dict:
                buffer = [key, data_dict[key]]
                obj_writer.writerow(buffer)

if __name__ == "__main__":
    # rewriteDataset()
    extractData()
    # showSpeed()