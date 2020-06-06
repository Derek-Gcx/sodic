import csv
import copy
import os


roadDict = {
    276183: 0,
    276184: 1,
    275911: 2, 
    275912: 3,
    276240: 4,
    276241: 5,
    276264: 6,
    276265: 7,
    276268: 8,
    276269: 9,
    276737: 10,
    276738: 11
}


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
    for index in range(12):
        group = [0]
        all_trajs.append(group)
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
                group = roadDict.get(int(road_id), -1)
                date = line[3].split(" ")[0]
                TTI = []
                TTI.append(line[1])
            else:
                if (len(TTI) == 6):
                    TTI.append(line[1])
                    all_trajs[group].append(copy.deepcopy(TTI))
                    if(all_trajs[group][0]==0):
                        all_trajs[group].pop(0)
                    TTI.pop(0)
                else:
                    TTI.append(line[1])
        for index in range(12):
            if (len(all_trajs[index]) == batch_size) or (group != index and len(all_trajs[index]) > 1):
                with open("./train/processed/kr"+str(index)+".csv", "a+", newline='') as objfile:
                    obj_writer = csv.writer(objfile)
                    for item in all_trajs[index]:
                        obj_writer.writerow(item)
                all_trajs[index] = [0]


def test_dp_withgroup():
    all_trajs = []
    for index in range(12):
        group = [0]
        all_trajs.append(group)
    traj = []
    count = -1
    group = 0
    road_id = 0
    for line in open("./train/toPredict_train_TTI.csv"):
        line = line.split(",")
        count += 1
        if (line[0] == "id_road"):
            continue
        else:
            if (road_id != line[0]):
                road_id = line[0]
                group = roadDict.get(int(road_id), -1)
                traj = []
                traj.append(line[1])
            if (len(traj) == 6):
                assert (count % 6 == 0)
                traj.append(0)
                all_trajs[group].append(copy.deepcopy(traj))
                if(all_trajs[group][0]==0):
                    all_trajs[group].pop(0)
                traj = []
                traj.append(line[1])
            else:
                traj.append(line[1])
    if (len(traj) == 6):
        traj.append(0)
        all_trajs[group].append(traj)
    for group in range(12):
        with open("./train/processed/ToPredict"+str(group)+".csv", "a+", newline='') as objfile:
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
    for line in open("./train/toPredict_train_TTI.csv"):
        line = line.split(",")
        if line[0] == 'id_road':
            continue
        else:
            if (road_id != line[0]):
                road_id = line[0]
                index = roadDict.get(int(road_id), -1)
                TTI = []
                TTI.append(line[1])
            else:
                if (len(TTI) == 5):
                    TTI.append(line[1])
                    TTI.append(0)
                    all_trajs[index].append(copy.deepcopy(TTI))
                    TTI.pop(-1)
                    if(all_trajs[index][0] == 0):
                        all_trajs[index].pop(0)
                    TTI.pop(0)
                else:
                    TTI.append(line[1])
    with open("./train/processed/ToPredict.csv", "a+", newline='') as objfile:
        obj_writer = csv.writer(objfile)
        while(len(all_trajs[0])>=120):
            for i in range(12):
                for item in all_trajs[i][:120]:
                    obj_writer.writerow(item)
                if(len(all_trajs[i]) >= 120):
                    all_trajs[i] = all_trajs[i][120:]
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
    # if(os.path.exists("D:\projects\python\sodic\train\processed\kr.csv")):
    os.remove(r"D:\projects\python\sodic\train\processed\kr.csv")
    # if(os.path.exists("D:\projects\python\sodic\train\processed\ToPredict.csv")):
    os.remove(r"D:\projects\python\sodic\train\processed\ToPredict.csv")
    # if(os.path.exists("D:\projects\python\sodic\train\submit.csv")):
    # os.remove(r"D:\projects\python\sodic\train\submit.csv")


if __name__ == '__main__':
    # clear()
    # train_dp_withgroup()
    test_dp()