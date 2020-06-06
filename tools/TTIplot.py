import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
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
            traj.append(float(line[1]))
    all_traj.append(traj)
    for i in range(len(all_traj)):
        plt.plot(all_traj[i], label=str(all_id[i]))
        # sns.distplot(all_traj[i], label=str(all_id[i]))
    plt.legend()
    plt.show()
        # plt.savefig('./pic/'+str(all_id[i])+'TTI.png')
        # plt.clf()