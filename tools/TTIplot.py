import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def trace_peak():
    """
    watch out the day with the biggest TTI
    """
    peak = 0
    next_peak = 0
    date = 0
    for line in open("./train/train_TTI.csv"):
        line = line.split(",")
        if line[0] == "id_road":
            continue
        if float(line[1]) > peak:
            peak = float(line[1])
        else:
            if(float(line[1]) < peak) and ((float(line[1])) > next_peak):
                next_peak = float(line[1])
                date = line[3].split(" ")[0]
    ID = ''
    ids = {}
    trajs = []
    traj = []
    for line in open("./train/train_TTI.csv"):
        line = line.split(",")
        if line[3].split(" ")[0] == date:
            if ID != line[0]:
                ID = line[0]
                if (traj != []):
                    trajs.append(traj)
                ids[len(trajs)] = line[0]
                traj = []
                traj.append(float(line[2]))
            else:
                traj.append(float(line[2]))
    if (traj != []):
        trajs.append(traj)
    data = pd.DataFrame(trajs).transpose()
    data.rename(columns=ids, inplace=True)
    sns.lineplot(data=data, dashes=False)
    plt.show()


def trace_speed():
    """
    draw the displot of speeds
    """
    all_id = []
    all_traj = []
    traj = []
    road_id = 0
    for line in open("./train/train_TTI.csv"):
        line = line.split(",")
        if line[0] == 'id_road':
            continue
        else:
            if (int(line[0]) != road_id):
                road_id = int(line[0])
                all_id.append(road_id)
                if traj != []:
                    all_traj.append(traj)
                traj = []
            traj.append(float(line[2]))
    all_traj.append(traj)
    for i in range(len(all_traj)):
        sns.distplot(all_traj[i], label=str(all_id[i]))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # trace_peak()
    # trace_speed()