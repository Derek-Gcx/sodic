import time
import json
import csv
import matplotlib.pyplot as plt
import math

module_name = "TrajClassify"

# 一次性向新的csv中写入的数据条目数
batch_size = 200


roadDict = {
    # x1 ------------ x2
    #   |            |
    #   |            |
    # x4 ------------ x3
    276183: [[114.01623,22.587799], [114.028397,22.591564], [114.028225,22.590315], [114.015651,22.586353], 90],  # 南坪快速路西东
    276184: [[114.01623,22.587799], [114.028397,22.591564], [114.028225,22.590315], [114.015651,22.586353], 270],  # 南坪快速路东西
    275911: [[114.015415,22.603272], [114.017603,22.604143], [114.016702,22.588136], [114.013655,22.587938], 0],  # 福龙路南北
    275912: [[114.015415,22.603272], [114.017603,22.604143], [114.016702,22.588136], [114.013655,22.587938], 180], # 福龙路北南,
    276240: [[114.015651,22.6065], [114.025993,22.611908], [114.027002,22.610819], [114.016831,22.60345], 60], # 留仙大道西东
    276241: [[114.015651,22.6065], [114.025993,22.611908], [114.027002,22.610819], [114.016831,22.60345], 240], # 留仙大道东西
    276264: [[114.026465,22.605688], [114.032087,22.605688], [114.032495,22.592574], [114.026573,22.591861], 0], # 玉龙路南北
    276265: [[114.026465,22.605688], [114.032087,22.605688], [114.032495,22.592574], [114.026573,22.591861], 180], # 玉龙路北南
    276268: [[114.021938,22.616702], [114.023247,22.617375], [114.035156,22.605233], [114.033976,22.604341], 315], # 新区大道南北
    276269: [[114.03404,22.604262], [114.034877,22.605074], [114.040306,22.599627], [114.039576,22.598874], 135], # 新区大道北南
    276737: [[114.023247,22.609274], [114.024663,22.609769], [114.028139,22.606322], [114.027131,22.605708], 315], # 致远中路北南
    276738: [[114.023247,22.609274], [114.024663,22.609769], [114.028139,22.606322], [114.027131,22.605708], 135], # 致远中路南北
}

def getRoadID(lng, lat, direction):
    # print("get position at {}, {}, direction {}".format(lng, lat, direction))
    pos = [lng, lat]
    ret = []

    for roadID, cors in roadDict.items():
        ok = True
        for i in range(4):
            cor1 = cors[i]
            cor2 = cors[(i+1)%4]
            vec1 = [cor2[0]-cor1[0], cor2[1]-cor1[1]]
            vec2 = [pos[0]-cor1[0], pos[1]-cor1[1]]

            outer_prod = vec1[0]*vec2[1] - vec1[1]*vec2[0]

            if outer_prod >= 0:
                ok = False
                break
        
        if ok == False:
            continue
        else:
            if abs(direction-cors[4]) < 90:
                ret.append(roadID)

    return ret


def rewrite_dataset():
    for line in open("./train/toPredict_train_gps.csv"):
    
        traj = []

        illegal_char = ["[", "\"", "]"]
        for char in illegal_char:
            line = line.replace(char, "")
        gps_records = line.split(",")[2:]

        for record in gps_records:
            lng, lat, speed, direction, seconds = [eval(i) for i in record.strip().split(" ")]

            clock = tuple(time.gmtime(seconds))[0:6]
            road_id = getRoadID(lng, lat, direction)
            if road_id == []:

                continue 

            # 写入数据entry的结构: [路的ID, 经度, 纬度, 方向, 速度, 时间(年, 月, 日, 时, 分, 秒)]
            traj.append([road_id, lng, lat, direction, speed, clock])


            if len(traj) == batch_size:
                # plt.scatter([i[1] for i in traj], [i[2] for i in traj], c="g")
                
                with open("./train/processed/gcx.csv", "a+", newline='') as objfile:
                    obj_writer = csv.writer(objfile)
                    obj_writer.writerows(traj)
                traj = []
                plt.show()
                # assert False

def bd_to_gcj():
    for road_id in roadDict.keys():
        
        for i in range(4):
            x_pi = 3.14159265358979324 * 3000.0 / 180.0;
            x = roadDict[road_id][i][0] - 0.0065;
            y = roadDict[road_id][i][1] - 0.006;
            z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi);
            theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi);
            roadDict[road_id][i][0] = z * math.cos(theta)
            roadDict[road_id][i][1] = z * math.sin(theta)

def show_trace():
    line_counter = 0
    useless_x = list()
    useless_y = list()
    useful_x = list()
    useful_y = list()
    useless_counter = 0
    for line in open("./train/toPredict_train_gps.csv"):
        line_counter += 1

        illegal_char = ["[", "\"", "]"]
        for char in illegal_char:
            line = line.replace(char, "")
        gps_records = line.split(",")[2:]

        for record in gps_records:
            lng, lat, speed, direction, seconds = [eval(i) for i in record.strip().split(" ")]
            road_id = getRoadID(lng, lat, direction)
            if road_id == []:
                useless_x.append(lng)
                useless_y.append(lat)
            else:
                # if road_id == 276183 or road_id == 276184:
                useful_x.append(lng)
                useful_y.append(lat)
                # else:
                #     useless_x.append(lng)
                #     useless_y.append(lat)
            
            if len(useless_x) == 4000:
                useless_counter += 4000
                print("has useless track", useless_counter)
                plt.scatter(useless_x, useless_y, s=3, c="r")
                useless_x = list()
                useless_y = list()
                if useless_counter >= 100000:
                    plt.show()
                    assert False
            if len(useful_x) == 5000:
                plt.scatter(useless_x, useless_y, s=3, c="r")
                useless_x = list()
                useless_y = list()

                plt.scatter(useful_x, useful_y, s=3, c="g")
                useful_x = list()
                useful_y = list()
                plt.show()
                assert False

                

if __name__ == "__main__" :
    # bd_to_gcj()
    # print(roadDict)
    # assert False
    show_trace()