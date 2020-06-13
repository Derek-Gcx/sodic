import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def loadData(path):
    raw_train = pd.DataFrame(pd.read_csv(path, header=None))
    x = raw_train.iloc[:, 0:6].values
    y = raw_train.iloc[:, 6:7].values
    return train_test_split(x, y, test_size=0.2)


def train(x_train, x_valid, y_train, y_valid):
    fixed_paras = {'n_estimators': 2000, 'learning_rate': 0.05,  'max_depth': 5, 'seed': 0,'min_child_weight':2, 
    'gamma':0.7, 'colsample_bytree': 0.7, 'reg_alpha': 2, 'reg_lambda': 1, 'subsample': 0.6}
    model = xgb.XGBRegressor(objective='reg:squarederror', **fixed_paras)
    model.fit(x_train, y_train)
    ans = model.predict(x_valid)
    print(sum(abs(ans - y_valid[0]))/len(x_valid))
    # optimized = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=8)
    # optimized.fit(x_train, y_train)
    # evalute_result = optimized.cv_results_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized.best_params_))
    # print('最佳模型得分:{0}'.format(optimized.best_score_))


def run(groups):
    for group in range(groups):
        x_train, x_valid, y_train, y_valid = loadData("./train/processed/kr"+str(group)+".csv")
        train(x_train, x_valid, y_train, y_valid)


if __name__ == '__main__':
    run(1)