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
        rawX = rawData.iloc[:, 0:6].values
        rawY = rawData.iloc[:, 6:7].values
        self.size = len(rawX)
        self.X = torch.tensor(rawX)
        self.Y = torch.tensor(rawY)

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
        self.gru1 = nn.GRU(input_size, 16, 1, dropout=0.2).double()
        self.gru2 = nn.GRU(16, 32, 1, dropout=0.2, bidirectional=True).double()
        mlp_list = [
            nn.Linear(32, 8),
            nn.Linear(8, output_size)
        ]
        self.mlp = nn.Sequential(*mlp_list).double()

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
                m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal(param)

    def forward(self, all_traj):
        # lstm_out, (h, c) = self.lstm(torch.transpose(all_traj, 0, 1).double())
        traj = torch.transpose(all_traj, 0, 1)
        out, h = self.gru1(traj.double())
        out, h = self.gru2(out.double())
        h = torch.sum(h, 0).reshape(1, h.shape[1], -1)
        z = self.mlp(torch.transpose(h, 0, 1))
        return z


group = "0"
input_size = 1
output_size = 1
batch_size = 1024
dataset = Dataset("./train/processed/kr"+group+".csv")
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train, valid = torch.utils.data.random_split(dataset, [train_size, valid_size])
whole_data = DataLoader(dataset=dataset, batch_size=batch_size)
train_data = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
valid_data = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True)
net = naive_LSTM(input_size, output_size, True)
para = net.parameters()
# optimizer = optim.Adam(para, lr=1e-03, betas=(0.9, 0.999))
optimizer = optim.RMSprop(para, lr=1e-03, momentum=0.9)
vis = visdom.Visdom(env='naive_LSTM')
vis.line([[0.]], [0],win='train',opts=dict(title='losses', legend=['loss']))


def train(epoches):
    for epoch in range(epoches):
        net.train()
        for batch_idx, (X, Y) in enumerate(train_data):
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
        if epoch == 99:
            torch.save(net.state_dict(), './out/group_'+group+'_LSTM_100.pth')


def valid():
    result = []
    real_val = []
    pre_val = []
    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(valid_data):
            X = X.reshape(X.shape[0], 6, input_size).double()
            Y = Y.reshape(Y.shape[0], -1, output_size).double()
            predi = net(X) 
            loss_fn = torch.nn.L1Loss(reduce='mean')
            loss = loss_fn(predi, Y)
            result.append(loss.item())
            Y = Y.reshape(Y.shape[0], 1).numpy().tolist()
            predi = net(X).reshape(X.shape[0], 1).numpy().tolist()
            real_val += Y
            pre_val += predi
    print("performance on validation set is {}".format(sum(result)/len(result)))
    plt.plot(real_val, label="real")
    plt.plot(pre_val, label="predi")
    plt.legend()
    plt.show()


def group_switch(road_id):
    g = str(gp.igmap(road_id))
    net.load_state_dict(torch.load('./out/group_'+g+'_LSTM_100.pth'))


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
    result = []
    indi = 0
    last_id = -1
    for line in open("./train/processed/ToPredict.csv"):
        line = line.split(",")[:6]
        road_id = int(road_ids[indi])
        if(last_id != road_id):
            last_id = road_id
        group_switch(road_id)
        indi += 1
        for index in range(len(line)):
            line[index] = float(line[index])
        for i in range(3):
            ipt_tensor = torch.from_numpy(np.array(line)).reshape(1, 6, output_size).double()
            predi = net(ipt_tensor).item()
            result.append(predi)
            line.pop(0)
            line.append(predi)
    with open("./train/submit.csv", "a+", newline='') as objfile:
        obj_writer = csv.writer(objfile)
        obj_writer.writerow(["id_sample", "TTI"])
        for i in range(len(result)):
            row = [i, result[i]]
            obj_writer.writerow(row)


if __name__ == "__main__":
    train(100)
    net.load_state_dict(torch.load('./out/group_'+group+'_LSTM_100.pth'))
    # valid()
    test()
    # boost_test()
