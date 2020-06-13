import csv
import pandas as pd
import os
import sys
"""
    devide_train_TTI_by_roadid():   将train_TTI按照道路分组， 输出到train/processed/devided_train_TTI目录下
    check_and_fill_nan():           对按道路分割后的数据集文件进行日期补全。也就是在2019-01-01 00：00：00～2019-03-31 23：50：00
                                        和2019-10-01 00：00：00～2019-12-21 23：50：00这两段时间内，缺少的数据记录将会用NAN补全。
    prepare_train_set_group12():    对于每条道路， 准备好网络的输入向量。 具体的entry形式为：过去六个时段的TTI+待预测的TTI；文件输出目录为train/processed/to_train
    add_neighbour_info():           向上一步生成的12个训练集进一步添加近邻信息, 输出目录同上
    drop_nan():                     删除训练集中含有nan的行
"""

sys.path.append(os.getcwd())

gp_num = 12
road_ids = [276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]
neighbour_info = {
    276183: [276184, 275911, 275912, 276264, 276265], 
    276184: [276183, 275911, 275912, 276264, 276265], 
    275911: [275912, 276183, 276184, 276240, 276241], 
    275912: [275911, 276183, 276184, 276240, 276241], 
    276240: [276241, 275911, 275912, 276737, 276738, 276268, 276269], 
    276241: [276240, 275911, 275912, 276737, 276738, 276268, 276269], 
    276264: [276265, 276183, 276184, 276737, 276738], 
    276265: [276264, 276183, 276184, 276737, 276738], 
    276268: [276269, 276240, 276241], 
    276269: [276268, 276240, 276241], 
    276737: [276738, 276240, 276241, 276264, 276265], 
    276738: [276737, 276240, 276241, 276264, 276265]
}

def devide_train_TTI_by_roadid():
    print("Start to devide train_TTI by road id.")

    entries = []
    df = pd.read_csv("./asset/train_TTI.csv")
    # date = df["time"].apply(lambda x: x.split(" ")[0])
    # clock = df["time"].apply(lambda x: x.split(" ")[1])
    # df = df.drop("time", axis=1)
    # df["date"] = date
    # df["time"] = clock

    for road_id in set(df["id_road"]):
        save_csv = df[df["id_road"] == road_id]
        save_csv.to_csv("./train/processed/devided_train_TTI/"+str(road_id)+".csv", header=True, index=False)

    print("Finish deviding train_TTI by road id.")

def check_and_fill_nan():
    print("Start to check and fill nan.")
    date_range1 = pd.date_range(start="2019-01-01 00:00:00", end="2019-03-31 23:50:00", freq="10T")
    date_range2 = pd.date_range(start="2019-10-01 00:00:00", end="2019-12-21 23:50:00", freq="10T")
    
    for road_id in road_ids:
        df = pd.read_csv("./train/processed/devided_train_TTI/"+str(road_id)+".csv")
        df = df.set_index("time")
        df = df.set_index(pd.to_datetime(df.index))

        new_df1 = df.reindex(date_range1, fill_value=None)
        new_df2 = df.reindex(date_range2, fill_value=None)
        new_df = new_df1.append(new_df2)

        new_df.to_csv("./train/processed/devided_train_TTI/"+str(road_id)+".csv", header=True, index=True)
    print("Finish filling nan.")

def prepare_train_set_group12():
    print("Start to prepare training set.")
    devided_csv_path = "./train/processed/devided_train_TTI/"

    entries = []

    for road_id in road_ids:
        print("\tAdd {} training set.".format(road_id))
        df1 = pd.read_csv(devided_csv_path+str(road_id)+".csv")
        total = df1.shape[0]
        endpoint = 7
        while endpoint <= total:
            TTIs = list(df1.iloc[endpoint-7:endpoint]["TTI"])
            entries.append(TTIs)
            endpoint += 1
        save_csv = pd.DataFrame(data=entries)
        save_csv.to_csv("./train/processed/to_train/train_"+str(road_id)+".csv", mode="w", header=False, index=False)
        entries = []

    print("Finish preparing training set.")

def add_neighbour_info():
    
    print("Start to add neighbour info.")
    train_csv_path = "./train/processed/to_train/"
    info_csv_path = "./train/processed/devided_train_TTI/"
    for road_id in road_ids:
        print("Adding neighbour info into", road_id)
        df1 = pd.read_csv(train_csv_path+"train_"+str(road_id)+".csv", index_col=False, header=None)
        neighbour_dfs = [pd.read_csv(train_csv_path+"train_"+str(i)+".csv", index_col=False, header=None) for i in neighbour_info[road_id]]
        neighbour_cnt = 0
        for neighbour_df in neighbour_dfs:
            TTIs = neighbour_df.iloc[:, 0:6]
            df1 = pd.concat([df1, TTIs], axis=1)
            neighbour_cnt += 1
        pred = df1.pop(6)
        df1["pred"] = pred

        df1.to_csv(train_csv_path+"train_"+str(road_id)+".csv", mode="w", index=False, header=False)

    print("Finish adding neighbour info.")

def drop_nan():
    print("Start to drop nan.")
    train_csv_path = "./train/processed/to_train/"
    for road_id in road_ids:
        df1 = pd.read_csv(train_csv_path+"train_"+str(road_id)+".csv", index_col=False, header=None)
        df1 = df1.dropna(axis=0)
        df1.to_csv(train_csv_path+"train_"+str(road_id)+".csv", mode="w", index=False, header=False)
    print("Finish dropping nan.")


if __name__=="__main__":
    # devide_train_TTI_by_roadid()
    # check_and_fill_nan()
    # prepare_train_set_group12()
    # add_neighbour_info()
    drop_nan()