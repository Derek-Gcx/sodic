import csv
import pandas as pd
import os
import sys
import copy
"""
    devide_train_TTI_by_roadid():   将train_TTI按照道路分组， 输出到train/processed/devided_train_TTI目录下
    check_and_fill_nan():           对按道路分割后的数据集文件进行日期补全。也就是在2019-01-01 00：00：00～2019-03-31 23：50：00
                                        和2019-10-01 00：00：00～2019-12-21 23：50：00这两段时间内，缺少的数据记录将会用NAN补全。
    prepare_train_set_group12():    对于每条道路， 准备好网络的输入向量。 具体的entry形式为：过去六个时段的TTI+待预测的TTI；文件输出目录为train/processed/to_train
    add_neighbour_info():           向上一步生成的12个训练集进一步添加近邻信息, 输出目录同上
    drop_nan():                     删除训练集中含有nan的行
"""

sys.path.append(os.getcwd())

interested = [276737, 276738]

gp_num = 12
road_ids = [276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]
all_id = [276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]
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
feature_nums = {
    276183: 36, 276184: 36, 275911: 36, 275912: 36, 276240: 48, 276241: 48,  276264: 36, 276265: 36, 276268: 24, 276269: 24, 276737: 36, 276738: 36
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

def prepare_test_dataset():
    trainTTI = pd.read_csv("./test/toPredict_train_TTI.csv", index_col=None)

    for road_id in road_ids:
        save_csv = trainTTI[trainTTI["id_road"] == road_id]
        save_csv.to_csv("./test/processed/devided_train_TTI/"+str(road_id)+".csv", header=True, index=False)

def check_and_fill_nan_test():
    print("Start to check and fill nan in test dateset.")
    date1 = ["2019-12-21", "2019-12-22", "2019-12-23", "2019-12-24", "2019-12-25", "2019-12-26"]
    clock1 = ["07:30:00", "09:30:00", "11:30:00", "13:30:00", "15:30:00", "17:30:00", "19:30:00"]
    date2 = ["2019-12-27", "2019-12-28", "2019-12-29", "2019-12-30", "2019-12-31", "2020-01-01"]
    clock2 = ["08:00:00", "10:00:00", "12:00:00", "14:00:00", "16:00:00", "18:00:00", "20:00:00"]
    date_ranges = []
    for date in date1:
        for clock in clock1:
            tmp = pd.date_range(start=" ".join([date, clock]), periods=9, freq="10T")
            date_ranges.append(tmp)
    for date in date2:
        for clock in clock2:
            tmp = pd.date_range(start=" ".join([date, clock]), periods=9, freq="10T")
            date_ranges.append(tmp)
    print("date range generated:", date_ranges)

    for road_id in road_ids:
        df = pd.read_csv("./test/processed/devided_train_TTI/"+str(road_id)+".csv")
        df = df.set_index("time")
        df = df.set_index(pd.to_datetime(df.index))

        new_dfs = [df.reindex(i, fill_value=None) for i in date_ranges]
        new_df = pd.concat(new_dfs, axis=0)

        new_df.to_csv("./test/processed/devided_train_TTI/"+str(road_id)+".csv", header=True, index=True)
    print("Finish checking and filling nan into test dataset.")

def construct_test_dataset():
    print("Start to construct test data. May take 1 min.")
    nolabel = pd.read_csv("./test/toPredict_noLabel.csv", index_col=None)
    nolabel["time"] = pd.to_datetime(nolabel["time"])
    nolabel = nolabel.set_index("time")
    nolabel = nolabel.sort_index()
    
    all_csvs = {}
    entry = []
    entries = []
    for road_id in all_id:
        tmp = pd.read_csv("./test/processed/devided_train_TTI/"+str(road_id)+".csv", index_col=0)
        tmp = tmp.set_index(pd.to_datetime(tmp.index))
        all_csvs[road_id] = tmp
    for time, row in nolabel.iterrows():
        road_id = row["id_road"]
        neighbours = copy.deepcopy(neighbour_info[road_id])
        neighbours.insert(0, road_id)
        date_range = pd.date_range(end=str(time), periods=7, freq="10T")[0:-1]
        for neighbour in neighbours:
            tmp = all_csvs[neighbour].reindex(date_range, fill_value=None)["TTI"]
            tmp = list(tmp)
            entry += tmp
        entries.append(entry)
        entry = []

    test = pd.DataFrame(entries, index=nolabel.index)
    test = pd.concat([nolabel, test], axis=1)
    test.to_csv("./test/processed/test_data.csv", index=False, header=True)
    print("test data constructed.")

            




    # # trainTTI = trainTTI.set_index("time")
    # # trainTTI = trainTTI.set_index(pd.to_datetime(trainTTI.index))
    # trainTTI["time"] = pd.to_datetime(trainTTI["time"])
    # trainTTI = trainTTI.set_index(["time", "id_road"])

    # nolabel = nolabel.drop(["id_sample"], axis=1)
    # nolabel = nolabel.sort_index()

    # Topredict = copy.deepcopy(nolabel)
    # for index, row in nolabel.iterrows():
    #     date_range = pd.date_range(end=index, periods=6, freq="10T")
    #     entry = []
    #     neighbours = neighbour_info[row["id_road"]]
    #     neighbours.insert(0, row["id_road"])
    #     print(neighbours)
        
    #     new_df = pd.DataFrame()
    #     for neighbour in neighbours:
    #         tmp = trainTTI.reindex([date_range, row["id_road"]])



if __name__=="__main__":
    if interested==[]:
        pass
    else:
        road_ids = interested
    # devide_train_TTI_by_roadid()
    # check_and_fill_nan()
    # prepare_train_set_group12()
    add_neighbour_info()
    drop_nan()
    prepare_test_dataset()
    check_and_fill_nan_test()
    construct_test_dataset()