import time
import csv
import matplotlib.pyplot as plt

module_name = "TrajClassify"

# 一次性向新的csv中写入的数据条目数
batch_size = 200

roadDict = {
    # x1 ------------ x2
    #   |            |
    #   |            |
    # x4 ------------ x3
    276183: [(114.02242, 22.594901), (114.035176, 22.598237),
             (114.034888, 22.596019), (114.022114, 22.592765), 90],  # 南坪快速路西东
    276184: [(114.02242, 22.594901), (114.035176, 22.598237),
             (114.034888, 22.596019), (114.022114, 22.592765), 270],  # 南坪快速路东西
    275911: [(114.021593, 22.609765), (114.023929, 22.609932),
             (114.023246, 22.59415), (114.020911, 22.59415), 0],  # 福龙路南北
    275912: [(114.021593, 22.609765), (114.023929, 22.609932),
             (114.023246, 22.59415), (114.020911, 22.59415), 180],  # 福龙路北南,
    276240: [(114.02224, 22.612492), (114.033056, 22.618581),
             (114.03444, 22.617414), (114.023409, 22.609907), 60],  # 留仙大道西东
    276241: [(114.02224, 22.612492), (114.033056, 22.618581),
             (114.03444, 22.617414), (114.023409, 22.609907), 240],  # 留仙大道东西
    276264: [(114.033092, 22.611842), (114.038661, 22.612159),
             (114.038769, 22.598346), (114.033344, 22.598079), 0],  # 玉龙路南北
    276265: [(114.033092, 22.611842), (114.038661, 22.612159),
             (114.038769, 22.598346), (114.033344, 22.598079), 180],  # 玉龙路北南
    276268: [(114.028528, 22.623035), (114.029606, 22.623818),
             (114.041715, 22.610924), (114.040853, 22.610157), 315],  # 新区大道南北
    276269: [(114.040853, 22.610157), (114.041715, 22.610924),
             (114.047626, 22.604452), (114.046925, 22.603767), 135],  # 新区大道北南
    276737: [(114.029705, 22.615703), (114.03434, 22.61215),
             (114.03434, 22.61215), (114.033819, 22.611716), 315],  # 致远中路北南
    276738: [(114.029705, 22.615703), (114.03434, 22.61215),
             (114.03434, 22.61215), (114.033819, 22.611716), 135],  # 致远中路南北
}


def getRoadID(lng, lat, direction):
    # print("get position at {}, {}, direction {}".format(lng, lat, direction))
    pos = [lng, lat]
    ret = []

    for roadID, cors in roadDict.items():
        ok = True
        for i in range(4):
            cor1 = cors[i]
            cor2 = cors[(i + 1) % 4]
            vec1 = [cor2[0] - cor1[0], cor2[1] - cor1[1]]
            vec2 = [pos[0] - cor1[0], pos[1] - cor1[1]]

            outer_prod = vec1[0] * vec2[1] - vec1[1] * vec2[0]

            if outer_prod >= 0:
                ok = False
                break

        if ok == False:
            continue
        else:
            if abs(direction - cors[4]) < 90:
                ret.append(roadID)

    return ret


if __name__ == "__main__":

    for line in open("./train/toPredict_train_gps.csv"):

        traj = []

        illegal_char = ["[", "\"", "]"]
        for char in illegal_char:
            line = line.replace(char, "")
        gps_records = line.split(",")[2:]

        for record in gps_records:
            lng, lat, speed, direction, seconds = [
                eval(i) for i in record.strip().split(" ")
            ]

            clock = tuple(time.gmtime(seconds))[0:6]
            road_id = getRoadID(lng, lat, direction)
            if road_id == []:

                continue

            # 写入数据entry的结构: [路的ID, 经度, 纬度, 方向, 速度, 时间(年, 月, 日, 时, 分, 秒)]
            traj.append([road_id, lng, lat, direction, speed, clock])

            if len(traj) == batch_size:
                # plt.scatter([i[1] for i in traj], [i[2] for i in traj], c="g")

                with open("./train/processed/gcx.csv", "a+",
                          newline='') as objfile:
                    obj_writer = csv.writer(objfile)
                    obj_writer.writerows(traj)
                traj = []
                plt.show()
                # assert False
