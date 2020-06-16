import numpy as np
import pandas as pd


def account(data):
    print(data.info)
    return 0


def write_back():
    return 0


def classify():
    for i in range(1, 21):
        car_index = 0
        data = pd.DataFrame(
            columns=['road_id', 'time_index', 'car_index', 'speed'])
        for line in open("./train/processed/gps/201912/12" + str(i) + ".csv",
                         "r"):
            for char in ["(", ")", "[", "]", "\""]:
                line = line.replace(char, "")
            line = line.split(",")
            line = list(map(float, line))
            line = [line[i:i + 5] for i in range(0, len(line), 5)]
            for record in line:
                time_index = record[2] * 6 + int(record[3] / 10)
                data = data.append(
                    {
                        "road_id": int(record[0]),
                        "time_index": int(time_index),
                        "car_index": int(car_index),
                        'speed': record[1]
                    },
                    ignore_index=True)
            car_index += 1
        account(data)
    return 0


if __name__ == "__main__":
    classify()