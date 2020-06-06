import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import visdom
from torch.utils.data import DataLoader


class Dataset(Data.Dataset):
    def __init__(self, path):
        rawData = pd.DataFrame(pd.read_csv(path, header=None))
        rawX = rawData.iloc[:, 0:12].values
        rawY = rawData.iloc[:, 12:15].values
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
        self.lstm = nn.LSTM(input_size, 16).double()
        mlp_list = [
            nn.Linear(16, 8),
            nn.Linear(8, output_size)
        ]
        self.mlp = nn.Sequential(*mlp_list).double()

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
                m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_normal(param)

    def forward(self, all_traj):
        h0 = torch.randn(1, all_traj.shape[0], 16).double()
        c0 = torch.randn(1, all_traj.shape[0], 16).double()
        lstm_out, (h, c) = self.lstm(
            torch.transpose(all_traj, 0, 1).double(), (h0, c0))
        z = self.mlp(torch.transpose(h, 0, 1))
        return z


batch_size = 1024
dataset = Dataset("./train/processed/kr.csv")
print(len(dataset))
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train, valid = torch.utils.data.random_split(dataset, [train_size, valid_size])
train_data = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
valid_data = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True)
net = naive_LSTM(2, 3, True)
para = net.parameters()
optimizer = optim.Adam(para, lr=1e-03, betas=(0.9, 0.999))
vis = visdom.Visdom(env='naive_LSTM')
vis.line([[0.]], [0],win='train',opts=dict(title='losses', legend=['loss']))


def train(epoches):
    for epoch in range(epoches):
        net.train()
        for batch_idx, (X, Y) in enumerate(train_data):
            X = X.reshape(X.shape[0], 6, 2).double()
            Y = Y.reshape(Y.shape[0], -1, 3).double()
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
        if epoch % 9 == 0:
            torch.save(net.state_dict(),'./out/1024naive_LSTM_{:02d}.pth'.format(epoch+1))


def valid():
    result = []
    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(valid_data):
            X = X.reshape(X.shape[0], 6, 2).double()
            Y = Y.reshape(Y.shape[0], -1, 3).double()
            predi = net(X) 
            loss_fn = torch.nn.L1Loss(reduce='mean')
            loss = loss_fn(predi, Y)
            result.append(loss.item())
    print("performance on validation set is {}".format(sum(result)/len(result)))


def test():
    result = []
    test = Dataset("./train/processed/ToPredict.csv")
    test_data = DataLoader(dataset=test, batch_size=batch_size)
    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(test_data):
            X = X.reshape(X.shape[0], 6, 2).double()
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


if __name__ == "__main__":
    # train(100)
    net.load_state_dict(torch.load("./out/1024naive_LSTM_100.pth"))
    valid()
    test()
