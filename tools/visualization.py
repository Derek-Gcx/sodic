import matplotlib.pyplot as plt
import numpy
import time
import math
import pandas as pd

PATH = "./asset/train_TTI.csv"

color = {
    276183: [], 
    276184: [], 
    275911: [], 
    275912: [], 
    276240: [], 
    276241: [], 
    276264: [], 
    276265: [], 
    276268: [], 
    276269: [], 
    276737: [], 
    276738: [], 
}


xs = {
    276183: [], 
    276184: [], 
    275911: [], 
    275912: [], 
    276240: [], 
    276241: [], 
    276264: [], 
    276265: [], 
    276268: [], 
    276269: [], 
    276737: [], 
    276738: [], 
}
ys = {
    276183: [], 
    276184: [], 
    275911: [], 
    275912: [], 
    276240: [], 
    276241: [], 
    276264: [], 
    276265: [], 
    276268: [], 
    276269: [], 
    276737: [], 
    276738: [], 
}

roads = {
    276183: {}, 
    276184: {}, 
    275911: {}, 
    275912: {}, 
    276240: {}, 
    276241: {}, 
    276264: {}, 
    276265: {}, 
    276268: {}, 
    276269: {}, 
    276737: {}, 
    276738: {}, 
}
color = ['pink', 'b', 'r', 'yellow', 'gray', 'green', 'purple', 'orange', 'black', '#6B8E23', '#DB7093', '#F5DEB3']

def show_TTI_avgspeed():
    first = True
    for line in open(PATH):
        if first:
            first = False
            continue
        line = line.split(",")
        road_id = eval(line[0])
        TTI = eval(line[1])
        avgspeed = eval(line[2])
        xs[road_id].append(avgspeed)
        ys[road_id].append(TTI)

    color = ['pink', 'b', 'r', 'yellow', 'gray', 'green', 'purple', 'orange', 'black', '#6B8E23', '#DB7093', '#F5DEB3']

    for road_id in xs.keys():
        col = color.pop()
        print(sum(ys[road_id])/len(ys[road_id]))
        print(max(ys[road_id]))
        plt.scatter(xs[road_id], ys[road_id], s=3, c=col, label=str(road_id));
    plt.legend()
    plt.show()

    # plt.scatter(xs[276265], ys[276265], s=3, c="r")
    # plt.show()

def show_fix_x_TTI_distibution():

    with open(PATH, "r") as p:
        line = p.readline()
        line = p.readline()
        while line:
            # print(line)
            line = line.split(",")
            road_id = eval(line[0])
            TTI = eval(line[1])
            if road_id != 276184:
                line = p.readline()
                continue
            
            avg_speed = eval(line[2])

            if 10<=avg_speed<=11:
                time = (line[3].split(" ")[-1]).split(":")
                h = time[0]
                m = time[1]
                
                with open(PATH, "r") as pp:
                    l = pp.readline()
                    l = pp.readline()
                    while l:
                        l = l.split(",")
                        time1 = (l[3].split(" ")[-1]).split(":")
                        h1 = time1[0]
                        m1 = time1[1]
                        if h1==h and m1==m:
                            road_id1 = eval(l[0])
                            TTI1 = eval(l[1])
                            xs[road_id1].append(TTI)
                            ys[road_id1].append(TTI1)
                        l = pp.readline()
            line = p.readline()
    print(xs, ys)

    color = ['pink', 'b', 'r', '#FF7F50', '#00008B', '#E9967A', '#B22222', '#DAA520', '#90EE90', '#6B8E23', '#DB7093', '#F5DEB3']

    for road_id in xs.keys():
        col = color.pop()

        plt.scatter(xs[road_id], ys[road_id], s=3, c=col);
    plt.show()

def showSpeed():
    count = 0
    with open("./asset/20191201_20191220.csv", "r") as FP:
        line = FP.readline()

        while line:

            y1 = []
            y2 = []
            ok = 3

            for char in ["[", "]", "\""]:
                line = line.replace(char, "")
            gps_records = line.split(",")[2:]

            # 循环读取当前订单的每一条GPS记录,记录感兴趣路段的结果,加入到entry列表中
            for record in gps_records:
                lng, lat, speed, direction, seconds = [eval(i) for i in record.strip().split(" ")]
                print("speed", speed, "direction", direction)
                y1.append(speed)
                if ok==3:
                    y2.append(speed)
                else:
                    y2.append(y2[-1])
                ok = (ok+1)%4

            x = range(len(y1))

            print(sum(y1)/len(y1), sum(y2)/len(y2))
            print(s(y1, sum(y1)/len(y1)), s(y2, sum(y2)/len(y2)))
            plt.subplot(121)
            plt.bar(x, y1)
            plt.subplot(122)
            plt.bar(x, y2)

            plt.show()
            line = FP.readline()
        
def s(data, average):
    total = 0
    for value in data:
        total += (value - average) ** 2
    
    stddev = math.sqrt(total/len(data))
    return stddev

def show_correlation():
    # 以一月一号为例,观察各时段下各路段TTI之间的差别
    color = ['pink', 'b', 'r', '#FF7F50', '#00008B', '#E9967A', '#B22222', '#DAA520', '#90EE90', '#6B8E23', '#DB7093', '#F5DEB3']
    y = {}
    with open("./asset/train_TTI.csv", "r") as fp:
        current = 0
        line = fp.readline()
        line = fp.readline()

        while line:
            line = line.split(",")
            if current != eval(line[0]):
                current = eval(line[0])
            # print("current set to ", current)
                y[current] = []
            # print(line)
            if line[3].startswith("2019-01-05"):
                y[current].append(eval(line[1]))
            line = fp.readline()
    print(len(y.keys()), len(y[276183]))

    for r in y.keys():
        c = color.pop()
        plt.plot(range(len(y[r])), y[r], c=c)
    plt.show()
    
def foo():
    color = ['pink', 'b', 'r', '#FF7F50', '#00008B', '#E9967A', '#B22222', '#DAA520', '#90EE90', '#6B8E23', '#DB7093', '#F5DEB3']
    TTIs = {}
    avg_speeds = {}
    with open("./asset/train_TTI.csv", "r") as fp:
        line = fp.readline()
        line = fp.readline()
        print(line)
        current = 0
        while line:
            line = line.split(",")
            if current != eval(line[0]):
                current = eval(line[0])
                TTIs[current] = []
                avg_speeds[current] = []
            if line[3].startswith("2019-10-18"):
                TTIs[current].append(eval(line[1]))
                avg_speeds[current].append(eval(line[2]))
            line = fp.readline()

    with open("./train/201910_11/20191018.csv", "r") as fp:
        line = fp.readline()
        cnt = 0
        count = {
            276183: {}, 
            276184: {}, 
            275911: {}, 
            275912: {}, 
            276240: {}, 
            276241: {}, 
            276264: {}, 
            276265: {}, 
            276268: {}, 
            276269: {}, 
            276737: {}, 
            276738: {}, 
        }
        average_speed = {
            276183: {}, 
            276184: {}, 
            275911: {}, 
            275912: {}, 
            276240: {}, 
            276241: {}, 
            276264: {}, 
            276265: {}, 
            276268: {}, 
            276269: {}, 
            276737: {}, 
            276738: {}, 
        }
        
        while line:
            cnt += 1
            # print(line, type(line))
            all_GPS_records = eval(line)
            # print(all_GPS_records, type(all_GPS_records[0]))
            if type(all_GPS_records)!=tuple:
                line = fp.readline()
                continue
            all_GPS_records = [eval(i) for i in all_GPS_records]
            if len(all_GPS_records) <= 2:
                line = fp.readline()
                continue
            df = pd.DataFrame(data=all_GPS_records, columns=["road_id", "speed", "time"])
            time_stamp = []
            for record in all_GPS_records:
                clock = record[-1]
                time_stamp.append(clock[0]*6+clock[1]//10+1)
                # print(clock, clock[0]*6+clock[1]//10+1)
            df["time_stamp"] = time_stamp

            gp = df.groupby(["road_id", "time_stamp"])
            a = gp.agg({"speed": "mean"})
            for key, group in gp:
                # if key[1] <= 3:
                #     print("H")
                count[key[0]][key[1]] = count[key[0]].get(key[1], 0) + 1
                if key[1] not in average_speed[key[0]].keys():
                    average_speed[key[0]][key[1]] = []
                average_speed[key[0]][key[1]].append(a.loc[key, "speed"])
            
            # print(cnt)
            line = fp.readline()

    # r = 275911

    for r in count.keys():
        p = [i*50 for i in TTIs[r]]
        plt.plot(range(144), p, c="r")
        plt.plot(range(144), avg_speeds[r], c="g")

        # print(count)
        pp = [ count[r].get(i, 0) for i in range(144)]
        # print(pp)
        ppp = [sum(average_speed[r].get(i, [0])) / len(average_speed[r].get(i, [0]))*3.6 for i in range(144)]
        plt.plot(range(144),pp, c="b")
        plt.plot(range(144), ppp, c="purple")
        plt.show()


def show_seasonality():
    color = ['pink', 'b', 'r', '#FF7F50', '#00008B', '#E9967A', '#B22222', '#DAA520', '#90EE90', '#6B8E23', '#DB7093', '#F5DEB3']
    TTIs = {}
    avg_speeds = {}
    with open("./asset/train_TTI.csv", "r") as fp:
        line = fp.readline()
        line = fp.readline()
        print(line)
        current = 0
        while line:
            line = line.split(",")
            if current != eval(line[0]):
                current = eval(line[0])
                TTIs[current] = []
                avg_speeds[current] = []
            if line[3].endswith("10:40:00\n"):
                TTIs[current].append(eval(line[1]))
                avg_speeds[current].append(eval(line[2]))
            line = fp.readline()

    for r in TTIs.keys():
        c = color.pop()
        plt.scatter(range(len(TTIs[r])), TTIs[r], c=c, s=3)
    plt.show()

def show_avg_TTI_around_year():
    for id in roads.keys():
        for i in range(144):
            roads[id][i] = []

    with open("./asset/train_TTI.csv", "r") as fp:
        current = 0
        line = fp.readline()
        line = fp.readline()

        while line:
            line = line.split(",")
            road_id = eval(line[0])
            TTI = eval(line[1])
            clock = line[-1].split(" ")[-1].split(":")
            time_stamp = int(clock[0])*6 + int(clock[1]) // 10
            roads[road_id][time_stamp].append(TTI)

            line = fp.readline()
        
    for id in roads.keys():
        roads[id] = [sum(i)/len(i) for i in roads[id].values()]

    interested = [276738, 276737,276240, 276241, 276264, 276265]
    for id in roads.keys():
        if id in interested:
            c = color.pop()
            plt.scatter(range(144), roads[id], s=3, c=c, label = str(id))
    plt.legend()
    plt.show()
    # import csv
    # with open("./a.csv", "w") as fp:
    #     writer = csv.writer(fp)
    #     writer.writerows(roads.items())
    return roads






if __name__ == "__main__":

    # show_seasonality()
    # foo()
    # show_TTI_avgspeed()
    show_avg_TTI_around_year()


