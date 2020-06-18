import xgboost as xgb
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

reader1=pd.read_csv("D:/GitHub/sodic/train/train_TTI.csv")
reader2=pd.read_csv("D:/GitHub/sodic/train/toPredict_train_TTI.csv")
raw=pd.concat([reader1,reader2],axis=0).drop_duplicates()
id_lst=[276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]
df_lst=[]
#Add speed here
for i in range(0,12):
    df=raw[raw.id_road==id_lst[i]][['time','TTI']]
    df['time']=pd.to_datetime(df['time'])
    df=df[df.time>='2019-10-01 00:00:00']
    df_lst.append(df[['time','TTI']])

column_names=[]
for i in range(10,70,10):
    for j in range(1,13):
        column_names.append(str(j)+"_"+str(i))

result_lst=[[],[],[]]
reader3=pd.read_csv("D:/GitHub/sodic/train/toPredict_noLabel.csv")
for b in range(0,3):
    #prepare train_withlabel
    train_raw=df_lst[0]
    for j in range(1,12):
        train_raw=pd.merge(train_raw,df_lst[j],how='left',on='time')
        #print("prepare:",j,len(df_lst[j]))
    train_raw.columns=list(range(0,13))
    train_withlabel=train_raw
    for k in range(1,7):
        new=train_raw.shift(k).iloc[:,1:]
        train_withlabel=pd.concat([train_withlabel,new],axis=1)
    train_withlabel=train_withlabel.dropna()
    for i in range(1,13):
        X=train_withlabel.iloc[:,13:]
        X.columns=column_names
        y=train_withlabel.iloc[:,i:i+1]
        print("tt:",i,len(df_lst[i-1]))

        #prepare test    
        target=reader3[reader3.id_road==id_lst[i-1]]
        batch=target[target.id_sample%3==b]
        batch['time']=pd.to_datetime(batch['time'])
        batch_time=pd.DataFrame(batch.time,columns=['time'])
        train_withlabel.rename(columns={0:'time'},inplace=True) 
        time_table=pd.concat([train_withlabel.iloc[:,0:1],batch_time],axis=0).drop_duplicates()
        time_table.columns=['time']
        time_table=time_table.sort_values(by="time" , ascending=True)
        time_table.index=range(0,len(time_table))
        index=time_table[time_table['time'].isin(list(batch['time']))].index-1
        train_withlabel.rename(columns={0:'time'},inplace=True) 
        test_raw=train_withlabel[train_withlabel['time'].isin(list(time_table.iloc[index].time))]
        test=test_raw.iloc[:,1:73]
        test.columns=column_names
        print("test:",i,len(df_lst[i-1]))

        #train and predict
        params = {'n_estimators':50, 'learning_rate': 0.1,  'max_depth': 10, 'seed': 0,'min_child_weight':3, 
                    'gamma':0, 'colsample_bytree': 0.9,'subsample': 0.9}
        model = xgb.XGBRegressor(objective='reg:gamma', **params)
        model.fit(X, y, eval_metric="mae") 
        ans = model.predict(test)
        
        #update train
        tti=pd.DataFrame(ans,columns=['TTI'])
        batch_time.index=range(0,len(batch_time))
        generated=pd.concat([batch_time,tti],axis=1)
        df_lst[i-1]=df_lst[i-1].append(generated)
        
        print("update:",i,len(df_lst[i-1]))
        
        #store result
        result_lst[b].append(new)