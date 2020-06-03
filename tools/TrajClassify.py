import time
import requests
import json
import csv

roadDict = {
    '南坪快速路西东': 276183,
    '南坪快速路东西': 276184,
    '福龙路南北': 275911,
    '福龙路北南': 275912,
    '留仙大道西东': 276240,
    '留仙大道东西': 276241,
    '玉龙路南北': 276264,
    '玉龙路北南': 276265,
    "新区大道南北": 276268,
    '新区大道北南': 276269,
    '致远中路北南': 276737,
    '致远中路南北': 276738
}


def getRoadID(lng, lat, direction):
    ak = "3zjUzXMHNwkOcUfwXM8vryzHHxMMKgiL"
    url = "http://api.map.baidu.com/reverse_geocoding/v3/?ak={}&output=json&coordtype=wgs84ll&location={},{}".format(
        ak, lat, lng)
    res = requests.get(url)
    json_data = json.loads(res.text)
    if json_data['status'] == 0:
        road = json_data['result']['addressComponent']['street']
        # print(road)
        # TODO:检验方向
        return roadDict.get(road, 0)
    else:
        print("error!")
        assert (0)
    return 0


for line in open("./train/toPredict_train_gps.csv"):
    traj = []
    line = line.split(",")[2:]
    for step in line:
        step = step[1:]
        illegal_char = ['[', ']', '\n', '"']
        for char in illegal_char:
            step = step.replace(char, '')
        step = step.split(" ")
        processed = []
        roadid = getRoadID(float(step[0]), float(step[1]), float(step[3]))
        processed.append(roadid)
        processed.append(float(step[2]))
        clock = time.gmtime(int(step[4]))
        processed.append(time.gmtime(int(step[4])))
        # TODO:筛选需要的轨迹
        traj.append(processed)
    # TODO:对时间处理，xxx应该是2019-12-26-8:00-9:00的样子
    with open("./train/xxx.csv", 'w+', newline='') as objfile:
        obj_writer = csv.writer(objfile)
        obj_writer.writerow(traj)