import matplotlib.pyplot as plt
import numpy

PATH = "./asset/train_TTI.csv"

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

    color = ['pink', 'b', 'r', '#FF7F50', '#00008B', '#E9967A', '#B22222', '#DAA520', '#90EE90', '#6B8E23', '#DB7093', '#F5DEB3']

    for road_id in xs.keys():
        col = color.pop()
        print(sum(ys[road_id])/len(ys[road_id]))
        print(max(ys[road_id]))
        plt.scatter(xs[road_id], ys[road_id], s=3, c=col);
    
    plt.show()

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




if __name__ == "__main__":
    show_fix_x_TTI_distibution()

