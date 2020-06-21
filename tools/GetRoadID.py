import time
import json
import csv
import matplotlib.pyplot as plt
import math

module_name = "GetRoadID"

roadDict = {
    # x1 ------------ x2
    #   |            |
    #   |            |
    # x4 ------------ x3
    276183: [[114.01623,22.587799], [114.028397,22.591564], [114.028225,22.590315], [114.015651,22.586353], 90],  # 南坪快速路西东
    276184: [[114.01623,22.587799], [114.028397,22.591564], [114.028225,22.590315], [114.015651,22.586353], 270],  # 南坪快速路东西
    275911: [[114.015812,22.603579], [114.017603,22.604143], [114.016702,22.588136], [114.014224,22.588166], 0],  # 福龙路南北
    275912: [[114.015812,22.603579], [114.017603,22.604143], [114.016702,22.588136], [114.014224,22.588166], 180], # 福龙路北南,
    276240: [[114.015651,22.6065], [114.025993,22.611908], [114.027002,22.610819], [114.016831,22.60345], 60], # 留仙大道西东
    276241: [[114.015651,22.6065], [114.025993,22.611908], [114.027002,22.610819], [114.016831,22.60345], 240], # 留仙大道东西
    276264: [[114.026181,22.605936], [114.027962,22.604985], [114.032983,22.596764], [114.029679,22.592108], [114.027082,22.591732], 0], # 玉龙路南北
    276265: [[114.026181,22.605936], [114.027962,22.604985], [114.032983,22.596764], [114.029679,22.592108], [114.027082,22.591732], 180], # 玉龙路北南
    276268: [[114.022083,22.616682], [114.023247,22.617375], [114.035156,22.605233], [114.034142,22.604519], 315], # 新区大道南北
    276269: [[114.022083,22.616682], [114.023247,22.617375], [114.040306,22.599627], [114.039576,22.598874], 135], # 新区大道北南
    276737: [[114.023247,22.609274], [114.024663,22.609769], [114.028139,22.606322], [114.027131,22.605708], 135], # 致远中路北南
    276738: [[114.023247,22.609274], [114.024663,22.609769], [114.028139,22.606322], [114.027131,22.605708], 315], # 致远中路南北
}

def getRoadID(lng, lat, direction):
    # print("get position at {}, {}, direction {}".format(lng, lat, direction))
    pos = [lng, lat]
    ret = []
    for roadID, cors in roadDict.items():
        ok = True
        num_cors = len(cors)-1
        for i in range(num_cors):
            cor1 = cors[i]
            cor2 = cors[(i+1)%num_cors]
            vec1 = [cor2[0]-cor1[0], cor2[1]-cor1[1]]
            vec2 = [pos[0]-cor1[0], pos[1]-cor1[1]]

            outer_prod = vec1[0]*vec2[1] - vec1[1]*vec2[0]

            if outer_prod >= 0:
                ok = False
                break
        
        if ok == False:
            continue
        else:
            if 0<direction<cors[-1]-270 or cors[-1]-90 < direction < cors[-1]+90 or cors[-1]+270 < direction < 360:
                # ret.append(roadID)
                return roadID
    return 0

def show_trace():
    line_counter = 0
    useless_x = list()
    useless_y = list()
    useful_x = list()
    useful_y = list()
    useless_counter = 0
    for line in open("./test/toPredict_train_gps.csv"):
        line_counter += 1

        illegal_char = ["[", "\"", "]"]
        for char in illegal_char:
            line = line.replace(char, "")
        gps_records = line.split(",")[2:]

        for record in gps_records:
            lng, lat, speed, direction, seconds = [eval(i) for i in record.strip().split(" ")]
            road_id = getRoadID(lng, lat, direction)
            # print(road_id)
            if road_id == 0:
                useless_x.append(lng)
                useless_y.append(lat)
            else:
                if road_id == 276269 or road_id == 276268:
                    useful_x.append(lng)
                    useful_y.append(lat)
                else:
                    useless_x.append(lng)
                    useless_y.append(lat)

            if len(useless_x) == 4000:
                useless_counter += 4000
                print("has useless track", useless_counter)
                plt.scatter(useless_x, useless_y, s=3, c="r")
                useless_x = list()
                useless_y = list()
            if len(useful_x) == 1000:
                plt.scatter(useless_x, useless_y, s=3, c="r")
                useless_x = list()
                useless_y = list()

                plt.scatter(useful_x, useful_y, s=3, c="g")
                useful_x = list()
                useful_y = list()

                plt.show()
                assert False

                

if __name__ == "__main__" :
    show_trace()