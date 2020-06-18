import csv
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import visdom
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
import tools.group as gp


class Dataset(Data.Dataset):
    def __init__(self, path):
        rawData = pd.DataFrame(pd.read_csv(path, header=None))
        rawX = rawData.iloc[:, 0:42].values
        rawY = rawData.iloc[:, 42:].values
        self.size = len(rawX)
        self.X = torch.tensor(rawX, dtype=torch.double)
        self.Y = torch.tensor(rawY, dtype=torch.double)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # 自定义transform()对训练数据进行预处理
        return self.X[index], self.Y[index]


class naive_LSTM(nn.Module):
    def __init__(self, input_size, output_size, is_training=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.is_training = is_training
        # self.lstm = nn.LSTM(input_size, 16, 1).double()
        # self.gru1 = nn.GRU(input_size, 16, 1, dropout=0.2).double()
        self.gru = nn.GRU(input_size, 32, 2, dropout=0.2, bidirectional=True).double()
        nerus = [32, 64, 16, output_size]
        drop_rate = [0.1, 0.1, 0]
        mlp_list = []
        for i in range(len(nerus) - 1):
            mlp_i = nn.Sequential(
                nn.Linear(nerus[i], nerus[i+1]),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(nerus[i+1]),
                nn.Dropout(drop_rate[i])
            )
            mlp_list.append(mlp_i)
        # mlp_list = [
        #     nn.Linear(32, 8),
        #     nn.Linear(8, output_size)
        # ]
        self.mlp = nn.Sequential(*mlp_list).double()

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
                m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)

    def forward(self, all_traj):
        # lstm_out, (h, c) = self.lstm(torch.transpose(all_traj, 0, 1).double())
        traj = torch.transpose(all_traj, 0, 1)
        out, h = self.gru(traj.double())
        # out, h = self.gru2(out.double())
        h = torch.sum(h, 0).reshape(h.shape[1], -1)
        z = self.mlp(h)
        return z


def run(group_index):
    road_id = str(gp.iv_map(group_index))
    input_size = 7
    output_size = 3
    batch_size = 1024
    dataset = Dataset("./train/processed/train_feature/"+road_id+".csv")
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
    whole_data = DataLoader(dataset=dataset, batch_size=batch_size)
    train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_data = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
    net = naive_LSTM(input_size, output_size, True)
    para = net.parameters()
    optimizer = optim.Adam(para, lr=1e-03, betas=(0.9, 0.999))
    # optimizer = optim.RMSprop(para, lr=1e-03, momentum=0.9)
    vis = visdom.Visdom(env='naive_LSTM')
    vis.line([[0.]], [0],win='train', opts=dict(title='losses', legend=['loss']))
    bg = 0

    def train(epoches):
        for epoch in range(epoches):
            net.train()
            for batch_idx, (X, Y) in enumerate(train_data):
                X = X.reshape(X.shape[0], 6, input_size).double()
                Y = Y.reshape(Y.shape[0], output_size).double()
                predi = net(X)
                loss_fn = torch.nn.L1Loss(reduction='mean')
                # loss_out = torch.nn.L1Loss(reduction='mean')
                assert(predi.shape == Y.shape)
                # loss = nn.functional.mse_loss(predi, Y)
                loss = loss_fn(predi, Y)
                net.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    vis.line([loss.item()], [batch_idx + epoch * len(dataset)/batch_size], win='train', update='append')
                    print("batch: {}, loss {}".format(batch_idx + epoch * len(dataset)/batch_size, loss.item()))
            if epoch == 499:
                torch.save(net.state_dict(), './out/group_'+str(group_index)+'_LSTM_500.pth')

    def valid():
        result = []
        real_val = []
        pre_val = []
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(valid_data):
                X = X.reshape(X.shape[0], 6, input_size).double()
                Y = Y.reshape(Y.shape[0], output_size).double()
                predi = net(X) 
                loss_fn = torch.nn.L1Loss(reduce='mean')
                loss = loss_fn(predi, Y)
                result.append(loss.item())
                Y = Y.reshape(-1, 1).numpy().tolist()
                predi = net(X).reshape(-1, 1).numpy().tolist()
                real_val += Y
                pre_val += predi
        print("performance on validation set is {}".format(sum(result)/len(result)))
        plt.plot(real_val, label="real")
        plt.plot(pre_val, label="predi")
        plt.legend()
        plt.show()

    def group_switch(road_id):
        g = str(gp.igmap(road_id))
        net.load_state_dict(torch.load('./out/group_'+g+'_bag8.pth'))

    def test():
        result = []
        test = Dataset("./train/processed/ToPredict.csv")
        test_data = DataLoader(dataset=test, batch_size=batch_size)
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(test_data):
                X = X.reshape(X.shape[0], 6, output_size).double()
                predi = net(X).numpy().tolist()
                for item in predi:
                    for TTI in item[0]:
                        result.append(TTI)
        with open("./train/submit.csv", "a+", newline='') as objfile:
            obj_writer = csv.writer(objfile)
            obj_writer.writerow(["id_sample", "TTI"])
            for i in range(len(result)):
                row = [i, result[i]]
                obj_writer.writerow(row)

    def bag_switch(road_id, bag):
        g = str(gp.igmap(road_id))
        model = './out/group_'+g+'_bag'+str(bag)+'.pth'
        net.load_state_dict(torch.load(model))

    def boost_test():
        road_ids = []
        count = 0
        for line in open('./train/toPredict_noLabel.csv'):
            line = line.split(",")
            if line[0] == "id_sample":
                continue
            else:
                if count % 3 == 0:
                    road_ids.append(line[1])
                count += 1
        all_result = []
        for bag in range(10):
            result = []
            indi = 0
            last_id = -1
            for line in open("./train/processed/ToPredict.csv"):
                line = line.split(",")[:6]
                road_id = int(road_ids[indi])
                if(last_id != road_id):
                    last_id = road_id
                # if(bg == 1):
                #     for index in range(len(line)):
                #         line[index] = float(line[index])
                #     for i in range(3):
                #         predi = 0
                #         for bag in range(10):
                #             bag_switch(road_id, bag)
                #             ipt_tensor = torch.from_numpy(np.array(line)).reshape(1, 6, output_size).double()
                #             predi += net(ipt_tensor).item()
                #         result.append(predi/10)
                #         line.pop(0)
                #         line.append(predi)
                #     indi += 1
                # else:
                bag_switch(road_id, bag)
                for index in range(len(line)):
                    line[index] = float(line[index])
                for i in range(3):
                    ipt_tensor = torch.from_numpy(np.array(line)).reshape(1, 6, output_size).double()
                    predi = net(ipt_tensor).item()
                    result.append(predi)
                    line.pop(0)
                    line.append(predi)
                indi += 1
            all_result.append(result)
        result = []
        for i in range(len(all_result[0])):
            temp = []
            for j in range(10):
                temp.append(all_result[j][i])
            temp.sort()
            min_var = 100
            s_index = 0
            for k in range(5):
                if(np.var(np.array(temp[k:k+4])) < min_var):
                    min_var = np.var(np.array(temp[k:k+4]))
                    s_index = k
            temp = temp[s_index:s_index+4]
            result.append(float(sum(temp))/len(temp))
        with open("./train/submit.csv", "a+", newline='') as objfile:
            obj_writer = csv.writer(objfile)
            obj_writer.writerow(["id_sample", "TTI"])
            for i in range(len(result)):
                row = [i, result[i]]
                obj_writer.writerow(row)
    
    def bagging(bags, epoches):
        for bag in range(bags):
            # sample_size = int(0.8 * train_size)
            sample, _ = torch.utils.data.random_split(dataset, [train_size, valid_size])
            sample_data = DataLoader(dataset=sample, batch_size=batch_size, shuffle=True)
            for epoch in range(epoches):
                net.train()
                for batch_idx, (X, Y) in enumerate(sample_data):
                    X = X.reshape(X.shape[0], 6, input_size).double()
                    Y = Y.reshape(Y.shape[0], -1, output_size).double()
                    predi = net(X)
                    loss_fn = torch.nn.L1Loss(reduce='mean')
                    assert(predi.shape == Y.shape)
                    loss = loss_fn(predi, Y)
                    net.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if batch_idx % 10 == 0:
                        vis.line([loss.item()], [batch_idx + epoch * len(dataset)/batch_size], win='train', update='append')
                        print("batch: {}, loss {}".format(batch_idx + epoch * len(dataset)/batch_size, loss.item()))
            torch.save(net.state_dict(), './out/group_'+group+'_bag'+str(bag)+'.pth')

    train(500)
    # bagging(10, 100)
    net.load_state_dict(torch.load('./out/group_'+str(group_index)+'_LSTM_500.pth'))
    valid()
    # test()
    # if(group == "11"):
    #     boost_test()


if __name__ == "__main__":
    for i in range(12):
        run(i)