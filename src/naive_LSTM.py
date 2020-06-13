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
from torch.utils.data import DataLoader
# import visdom

sys.path.append(os.getcwd())

road_ids = [276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]
# interested = [276265, 276737, 276738]
interested = []
feature_nums = {
    276183: 36, 276184: 36, 275911: 36, 275912: 36, 276240: 48, 276241: 48,  276264: 36, 276265: 36, 276268: 24, 276269: 24, 276737: 36, 276738: 36
}

class Dataset(Data.Dataset):
    def __init__(self, path, road_id):
        rawData = pd.DataFrame(pd.read_csv(path, header=None))
        feature_num = feature_nums[road_id]
        
        rawX = rawData.iloc[:, 0:feature_num].values
        rawY = rawData.iloc[:, feature_num:feature_num+1].values
        self.size = len(rawX)
        self.X = torch.tensor(rawX, dtype=torch.double)
        self.Y = torch.tensor(rawY, dtype=torch.double)
        self.road_id = road_id

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class naive_LSTM(nn.Module):
    def __init__(self, input_size, output_size, road_id, is_training=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.is_training = is_training
        self.road_id = road_id

        self.gru1 = nn.GRU(input_size, 16, 1, dropout=0.2).double()
        self.gru2 = nn.GRU(16, 32, 1, dropout=0.2, bidirectional=True).double()
        mlp_list = [
            nn.Linear(32, 8), 
            nn.Linear(8, output_size)
        ]
        self.mlp = nn.Sequential(*mlp_list).double()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal(param)

    def forward(self, all_traj):
        traj = torch.transpose(all_traj, 0, 1)
        out, h = self.gru1(traj.double())
        out, h = self.gru2(out.double())
        h = torch.sum(h, 0).reshape(1, h.shape[1], -1)
        z = self.mlp(torch.transpose(h, 0, 1))
        return z


batch_size=1024

def prepare_dataset(road_id):
    dataset = Dataset("./train/processed/to_train/train_"+str(road_id)+".csv", road_id)
    train_size = int(0.8*len(dataset))
    valid_size = len(dataset) - train_size

    train, valid = torch.utils.data.random_split(dataset, [train_size, valid_size])

    whole_data = DataLoader(dataset=dataset, batch_size=batch_size)
    train_data = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    valid_data = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True)
    # train_data = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # valid_data = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return whole_data, train_data, valid_data



def train(net: naive_LSTM, train_data, epoches):
    optimizer = optim.RMSprop(net.parameters(), lr=1e-03, momentum=0.9)
    c = 0
    for epoch in range(epoches):
        net.train()
        for batch_idx, (X, Y) in enumerate(train_data):
            c+=1

            X = X.reshape(X.shape[0], 6, net.input_size).double()
            Y = Y.reshape(Y.shape[0], -1, net.output_size).double()

            pred = net(X)

            # TODO 讨论loss选用
            loss_fn = torch.nn.L1Loss(reduce="mean")
            # assert pred.shape == Y.shape

            loss = loss_fn(pred, Y)
            net.zero_grad()
            loss.backward()
            optimizer.step()

            if c % 10 == 0:
                # vis.line([loss.item()], [batch_idx + epoch * len(train_data)/batch_size], win='train', update='append')
                print("batch: {}, loss {}".format(c, loss.item()))

        if epoch == 99:
            # torch.save(net.state_dict(), "./out/"+str(net.road_id)+".pth")
            print("model for", net.road_id, "saved.")
    return net

def valid(net: naive_LSTM, valid_data):
    result = []
    real_val = []
    pre_val = []
    with torch.no_grad():
        for batch_idx, (X,Y) in enumerate(valid_data):
            X = X.reshape(X.shape[0], 6, net.input_size).double()
            Y = Y.reshape(Y.shape[0], -1, net.output_size).double()

            pred = net(X)
            loss_fn = torch.nn.L1Loss(reduce="mean")
            loss = loss_fn(pred, Y)

            result.append(loss.item())
            # TODO check
            Y = Y.reshape(Y.shape[0], 1).numpy().tolist()
            pred = pred.reshape(X.shape[0], 1).numpy().tolist()
            real_val += Y
            pre_val += pred
    print("performance on validation set is {}".format(sum(result)/len(result)))
    plt.plot(real_val, label="real")
    plt.plot(pre_val, label="pred")
    plt.legend
    plt.show()



if __name__ == "__main__":

    to_train_list = road_ids if interested==[] else interested
    print("Going to train", to_train_list)
    for road_id in to_train_list:
        input_size = feature_nums[road_id] // 6
        output_size = 1
        # vis = visdom.Visdom(env='naive_LSTM')
        # vis.line([[0.]], [0],win='train',opts=dict(title='losses', legend=['loss']))

        whole_data, train_data, valid_data = prepare_dataset(road_id)

        net = naive_LSTM(input_size, output_size, road_id, True)

        net = train(net, train_data, 100)

        valid(net, valid_data)


        
        