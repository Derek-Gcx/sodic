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
    """
    torch Dataset类，用来加载数据集
    """
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
        return self.X[index], self.Y[index]


class GRU(nn.Module):
    def __init__(self, input_size, output_size, is_training=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.is_training = is_training
        self.gru1 = nn.GRU(input_size, 16, 1, dropout=0.2).double()
        self.gru2 = nn.GRU(16, 64, 1, dropout=0.2, bidirectional=True).double()
        mlp_list = [
            nn.Linear(64, 32),
            nn.Linear(32, output_size)
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
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)

    def forward(self, all_traj):
        """
        前向传播
        """
        traj = torch.transpose(all_traj, 0, 1)
        out, h = self.gru1(traj.double())
        out, h = self.gru2(out.double())
        h = torch.sum(h, 0).reshape(h.shape[1], -1)
        z = self.mlp(h)
        return z


def run(road_index):
    # 定义全局变量
    road_id = str(gp.iv_map(road_index))
    input_size = 7
    output_size = 3
    batch_size = 1024
    # 加载数据集
    dataset = Dataset("./train/processed/train_feature/"+road_id+".csv")
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
    whole_data = DataLoader(dataset=dataset, batch_size=batch_size)
    train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_data = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
    # 初始化网络和优化器
    net = GRU(input_size, output_size, True)
    para = net.parameters()
    optimizer = optim.Adam(para, lr=1e-03, betas=(0.9, 0.999))
    # 定义visdom监视器
    vis = visdom.Visdom(env='GRU')
    vis.line([[0.]], [0], win='train', opts=dict(title='losses', legend=['loss']))

    def train(epoches):
        """
        训练当前road_index上的网络
        """
        for epoch in range(epoches):
            net.train()
            for batch_idx, (X, Y) in enumerate(train_data):
                X = X.reshape(X.shape[0], 6, input_size).double()
                Y = Y.reshape(Y.shape[0], output_size).double()
                predi = net(X)
                loss_fn = torch.nn.L1Loss(reduction='mean')
                assert(predi.shape == Y.shape)
                # loss = nn.functional.mse_loss(predi, Y)
                loss = loss_fn(predi, Y)
                net.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    vis.line([loss.item()], [batch_idx + epoch * len(dataset)/batch_size], win='train', update='append')
                    print("batch: {}, loss {}".format(batch_idx + epoch * len(dataset)/batch_size, loss.item()))
            if epoch == epoches-1:
                torch.save(net.state_dict(), './out/group_'+str(road_index)+'_GRU_100.pth')

    def valid():
        """
        查看验证集上的性能
        """
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

    def para_switch(road_id):
        """
        切换不同的road_index参数
        """
        g = str(gp.iimap(road_id))
        net.load_state_dict(torch.load('./out/group_'+g+'_GRU_100.pth'))

    def test():
        """
        输出最终测试集上的输出结果
        """
        with torch.no_grad():
            result = []
            count = 0
            id_count = 0
            ids = ["276183", "276184", "275911", "275912", "276240", "276241", "276264", "276265", "276268", "276269", "276737", "276738"]
            for line in open("././train/processed/test_feature/ToPredict.csv"):
                line = line.replace("\n", "").split(",")[:-3]
                line = list(map(float, line))
                if(count % 7 == 0):
                    para_switch(int(ids[id_count]))
                    id_count = (id_count+1) % len(ids)
                count += 1
                line += line
                ipt_tensor = torch.from_numpy(np.array(line)).reshape(2, 6, input_size).double()
                result += net(ipt_tensor)[0, :].reshape(-1).numpy().tolist()

            with open("./train/submit.csv", "a+", newline='') as objfile:
                obj_writer = csv.writer(objfile)
                obj_writer.writerow(["id_sample", "TTI"])
                for i in range(len(result)):
                    row = [i, result[i]]
                    obj_writer.writerow(row)

    def figure():
        """
        展示当前road_index整个数据集上拟合效果
        """
        result = []
        real_val = []
        pre_val = []
        for _ in range(3):
            real_val.append([])
            pre_val.append([])
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(whole_data):
                X = X.reshape(X.shape[0], 6, input_size).double()
                Y = Y.reshape(Y.shape[0], output_size).double()
                predi = net(X)
                loss_fn = torch.nn.L1Loss(reduce='mean')
                loss = loss_fn(predi, Y)
                result.append(loss.item())
                Y = np.array(Y.numpy().tolist())
                predi = np.array(net(X).numpy().tolist())
                for o in range(3):
                    real_val[o] += list(Y[:, o])
                    pre_val[o] += list(predi[:, o])
        print("performance on whole set is {}".format(sum(result)/len(result)))
        for o in range(3):
            plt.plot(real_val[o], label="real")
            plt.plot(pre_val[o], label="predi")
            plt.xlabel('time blocks',fontsize=14)
            plt.ylabel('TTI',fontsize=14)
            plt.legend()
            plt.show()

    train(100)
    net.load_state_dict(torch.load('./out/group_'+str(road_index)+'_GRU_100.pth'))
    valid()
    # figure()
    if road_index == 11:
        test()


if __name__ == "__main__":
    for i in range(12):
        run(i)
