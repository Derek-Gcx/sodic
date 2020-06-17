import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def loadData(path):
    raw_train = pd.DataFrame(pd.read_csv(path, header=None))
    x = raw_train.iloc[:, 1:7].values
    y = raw_train.iloc[:, 7:8].values
    return train_test_split(x, y, test_size=0.2)


def train(x_train, x_valid, y_train, y_valid):
    fixed_paras = {'n_estimators':49, 'learning_rate': 0.1,  'max_depth': 7, 'seed': 0,'min_child_weight':6, 
    'gamma':0, 'colsample_bytree': 0.9,'subsample': 0.9}
    model = xgb.XGBRegressor(objective='reg:squarederror', **fixed_paras)
    model.fit(x_train, y_train, eval_metric="mae") 
    ans = model.predict(x_valid)
    print(sum(abs(ans - y_valid[0]))/len(x_valid))
    feat_imp = pd.Series(model.feature_importances_, index=['TTI','speed','count','avg','var','margin']).sort_values(ascending=False)
    plt.figure(figsize=(16, 5))
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    # optimized = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=8)
    # optimized.fit(x_train, y_train)
    # evalute_result = optimized.cv_results_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    # print('参数的最佳取值：{0}'.format(optimized.best_params_))
    # print('最佳模型得分:{0}'.format(optimized.best_score_))


def run():
    x_train, x_valid, y_train, y_valid = loadData("./train/processed/gps/201912/275911.csv")
    train(x_train, x_valid, y_train, y_valid)


if __name__ == '__main__':
    run()