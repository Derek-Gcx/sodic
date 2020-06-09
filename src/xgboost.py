import xgboost as xgb
import pandas as pd

params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 0,
    'nthread': 4,
}

def train(groups):
    for group in groups:
        raw_data = pd.DataFrame(pd.read_csv("./train/processed/kr"+group+".csv", header=None))
        data = rawData.iloc[:, 0:6].values
        label = rawData.iloc[:, 6:7].values
        dtrain = xgb.DMatrix(data, label=label)
        plst = params.items()
        num_rounds = 500
        model = xgb.train(plst, dtrain, num_rounds)