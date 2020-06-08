import csv
import copy
import os
import sys
import re

sys.path.append(os.getcwd())
import tools.group as gp

gp_num = 12


def train_dp_withspeed():
    all_trajs = []
    traj = []
    TTI = []
    road_id = 0
    date = ""
    batch_size = 256
    for line in open("./train/train_TTI.csv"):
        line = line.split(",")
        if line[0] == 'id_road':
            continue
        else:
            if (road_id != line[0]) or (date != line[3].split(" ")[0]):
                road_id = line[0]
                date = line[3].split(" ")[0]
                traj = []
                traj.append(line[1])
                traj.append(line[2])
                TTI = []
                TTI.append(line[1])
            else:
                if (len(TTI) == 6):
                    TTI.append(line[1])
                    temp_traj = copy.deepcopy(traj[:12])
                    temp_traj.append(TTI[-1])
                    all_trajs.append(copy.deepcopy(temp_traj))
                    traj.append(line[1])
                    traj.append(line[2])
                    traj = traj[2:]
                    TTI.pop(0)
                else:
                    traj.append(line[1])
                    traj.append(line[2])
                    assert (float(traj[0]) < 20)
                    TTI.append(line[1])
        if len(all_trajs) == batch_size:
            with open("./train/processed/kr.csv", "a+", newline='') as objfile:
                obj_writer = csv.writer(objfile)
                for item in all_trajs:
                    obj_writer.writerow(item)
            all_trajs = []


def train_dp():
    all_trajs = []
    TTI = []
    road_id = 0
    date = ""
    batch_size = 256
    for line in open("./train/train_TTI.csv"):
        line = line.split(",")
        if line[0] == 'id_road':
            continue
        else:
            if (road_id != line[0]) or (date != line[3].split(" ")[0]):
                road_id = line[0]
                date = line[3].split(" ")[0]
                TTI = []
                TTI.append(line[1])
            else:
                if (len(TTI) == 6):
                    TTI.append(line[1])
                    all_trajs.append(copy.deepcopy(TTI))
                    TTI.pop(0)
                else:
                    TTI.append(line[1])
        if len(all_trajs) == batch_size:
            with open("./train/processed/kr.csv", "a+", newline='') as objfile:
                obj_writer = csv.writer(objfile)
                for item in all_trajs:
                    obj_writer.writerow(item)
            all_trajs = []


def train_dp_withgroup():
    all_trajs = []
    T = []
    count = []
    for index in range(gp_num):
        group = [0]
        all_trajs.append(group)
        T.append(0)
        count.append(0)
    TTI = []
    road_id = 0
    date = ""
    batch_size = 256
    group = 0
    for line in open("./train/train_TTI.csv"):
        line = line.split(",")
        if line[0] == 'id_road':
            continue
        else:
            if (road_id != line[0]) or (date != line[3].split(" ")[0]):
                road_id = line[0]
                group = gp.igmap(int(road_id))
                date = line[3].split(" ")[0]
                T[group] += float(line[1])
                count[group] += 1
                TTI = []
                TTI.append(line[1])
            else:
                if (len(TTI) == 6):
                    TTI.append(line[1])
                    all_trajs[group].append(copy.deepcopy(TTI))
                    if (all_trajs[group][0] == 0):
                        all_trajs[group].pop(0)
                    TTI.pop(0)
                else:
                    TTI.append(line[1])
    for index in range(gp_num):
        T[index] /= count[index]
        if (len(all_trajs[index]) == batch_size) or (
                group != index and len(all_trajs[index]) > 1):
            with open("./train/processed/kr" + str(index) + ".csv",
                      "a+",
                      newline='') as objfile:
                obj_writer = csv.writer(objfile)
                for item in all_trajs[index]:
                    for i in range(len(item)):
                        item[i] = float(item[i]) - T[index]
                    obj_writer.writerow(item)
            all_trajs[index] = [0]
            print(T[index])


def test_dp_withgroup():
    all_trajs = []
    for i in range(12):
        all_trajs.append([0])
    TTI = []
    road_id = 0
    index = -1
    count = 0
    for line in open("./train/toPredict_train_TTI.csv"):
        line = line.split(",")
        if line[0] == 'id_road':
            continue
        else:
            count += 1
            if (road_id != line[0]):
                road_id = line[0]
                index = gp.iimap(int(road_id))
                TTI = []
                TTI.append(line[1])
            else:
                if (len(TTI) == 5):
                    TTI.append(line[1])
                    TTI.append(0)
                    all_trajs[index].append(copy.deepcopy(TTI))
                    assert (count % 6 == 0)
                    TTI = []
                    if (all_trajs[index][0] == 0):
                        all_trajs[index].pop(0)
                else:
                    TTI.append(line[1])
    for group in range(12):
        with open("./train/processed/ToPredict" + str(group) + ".csv",
                  "a+",
                  newline='') as objfile:
            obj_writer = csv.writer(objfile)
            for item in all_trajs[group]:
                obj_writer.writerow(item)


def test_dp():
    all_trajs = []
    for i in range(12):
        all_trajs.append([0])
    TTI = []
    road_id = 0
    index = -1
    count = 0
    for line in open("./train/toPredict_train_TTI.csv"):
        line = line.split(",")
        if line[0] == 'id_road':
            continue
        else:
            count += 1
            if (road_id != line[0]):
                road_id = line[0]
                index = gp.iimap(int(road_id))
                TTI = []
                TTI.append(line[1])
            else:
                if (len(TTI) == 5):
                    TTI.append(line[1])
                    TTI.append(0)
                    all_trajs[index].append(copy.deepcopy(TTI))
                    assert (count % 6 == 0)
                    TTI = []
                    if (all_trajs[index][0] == 0):
                        all_trajs[index].pop(0)
                else:
                    TTI.append(line[1])
    with open("./train/processed/ToPredict.csv", "a+", newline='') as objfile:
        obj_writer = csv.writer(objfile)
        while (len(all_trajs[0]) >= 7):
            for i in range(12):
                for item in all_trajs[i][:7]:
                    obj_writer.writerow(item)
                if (len(all_trajs[i]) >= 7):
                    all_trajs[i] = all_trajs[i][7:]
                else:
                    all_trajs[i] = []


def test_dp_withspeed():
    all_trajs = []
    traj = []
    count = -1
    for line in open("./train/toPredict_train_TTI.csv"):
        line = line.split(",")
        if (line[0] == "id_road"):
            continue
        count += 1
        if (len(traj) == 12):
            assert (count % 6 == 0)
            traj.append(0)
            all_trajs.append(copy.deepcopy(traj))
            traj = []
            traj.append(line[1])
            traj.append(line[2])
        else:
            traj.append(line[1])
            traj.append(line[2])
    if (len(traj) == 12):
        all_trajs.append(traj)
    with open("./train/processed/ToPredict.csv", "a+", newline='') as objfile:
        obj_writer = csv.writer(objfile)
        for item in all_trajs:
            obj_writer.writerow(item)


def clear():
    pwd = os.getcwd() + r'\train\processed'
    for filename in os.listdir(pwd):
        if (re.search("kr", filename) != None):
            os.remove(pwd + "\\" + filename)
    # if(os.path.exists("D:\projects\python\sodic\train\processed\kr.csv")):
    # os.remove(r"D:\projects\python\sodic\train\processed\kr.csv")
    # if(os.path.exists("D:\projects\python\sodic\train\processed\ToPredict.csv")):
    # os.remove(r"D:\projects\python\sodic\train\processed\ToPredict.csv")
    # if(os.path.exists("D:\projects\python\sodic\train\submit.csv")):
    # os.remove(r"D:\projects\python\sodic\train\submit.csv")


if __name__ == '__main__':
    clear()
    train_dp_withgroup()
    test_dp()