import xgboost as xgb
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np

id_lst=[276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]
raw_lst=[]
for i in range(0,12):
    path1="./train/train_features/"+str(id_lst[i])+".csv"
    raw1=pd.read_csv(path1)
    path2="./train/test_features/"+str(id_lst[i])+".csv"
    raw2=pd.read_csv(path2).dropna()
    raw=pd.concat([raw1,raw2],axis=0).drop_duplicates()
    raw.columns=["time","id_road","TTI","speed" ,"avg","low_ratio" ,"time_block","day_of_week" ,"is_weekend"]
    raw=raw[raw["time"]>="2019-10-01 00:00:00"]
    raw_lst.append(raw)
df_lst=[]
for i in range(0,12):
    df=raw_lst[i]
    df.columns=["time_"+str(i),"id_road","TTI_"+str(i),"speed" ,"avg_"+str(i),"low_ratio" ,"time_block","day_of_week" ,"is_weekend"]
    df=df[["time_"+str(i),"TTI_"+str(i)]]
    df=df[df["time_"+str(i)]>="2019-10-01 00:00:00"]
    time_range=list(raw_lst[0]["time_0"])
    df= df.set_index("time_"+str(i))
    df= df.reindex(time_range, fill_value=None)
    df.fillna(method='ffill',inplace=True)
    df["time_"+str(i)]=df.index
    df.index=range(0,len(df))
    df_lst.append(df[["time_"+str(i),"TTI_"+str(i)]])

column_names=[]
for i in range(10,70,10):
    for j in range(1,13):
        column_names.append("tti_"+str(j)+"_"+str(i))
column_names.append("time_block")
column_names.append("is_weekend")

#prepare train_withlabel
train_raw=df_lst[0].sort_values(by="time_0" , ascending=True)
train_raw.index=range(0,len(train_raw))
for i in range(1,12):
    df=df_lst[i].sort_values(by="time_"+str(i),ascending=True)
    df.index=range(0,len(df))
    train_raw=pd.DataFrame(train_raw).join(df[["TTI_"+str(i)]])
train_withlabel=train_raw
for k in range(1,7):
    new=train_raw.shift(k).iloc[:,1:]
    train_withlabel=pd.concat([train_withlabel,new],axis=1)
train_withlabel=train_withlabel.dropna()
train_withlabel=train_withlabel.drop_duplicates(['time_0'])
train_withlabel.index=range(0,len(train_withlabel))
df=raw_lst[0][raw_lst[0]['time_0']>="2019-10-01 00:00:00"].shift(-6)
df.index=range(0,len(df))
new_feature=df[["time_block","is_weekend"]].dropna()
train_withlabel=train_withlabel.join(new_feature)

#train and predict
result_lst=[[],[],[]]
reader3=pd.read_csv("D:/GitHub/sodic/train/toPredict_noLabel.csv")
for b in range(0,3):
    for i in range(1,13):
        X=train_withlabel[train_withlabel["time_0"]<="2019-12-21 00:00:00"]
        X=X.iloc[:,13:]
        X.columns=column_names
        X=X.shift(b).dropna()
        
        y=train_withlabel[train_withlabel["time_0"]<="2019-12-21 00:00:00"]
        y=y.iloc[:,i:(i+1)]
        y=y.shift(-b).dropna()

        #prepare test     
        target=reader3[reader3.id_road==id_lst[i-1]]
        batch=target[target.id_sample%3==b]
        #print("batch:",len(batch))
        batch['time']=pd.to_datetime(batch['time'])
        train_withlabel['time_0']=pd.to_datetime(train_withlabel['time_0'])
        batch_time=pd.DataFrame(batch.time)
        batch_time.columns=["time_0"]
        time_table=pd.concat([train_withlabel.iloc[:,0:1],batch_time],axis=0).drop_duplicates()

        time_table=time_table.sort_values(by="time_0" ,ascending=True)
        time_table.index=range(0,len(time_table))
        index=time_table[time_table["time_0"].isin(list(batch["time"]))].index-(b+1)
        test_raw=train_withlabel[train_withlabel['time_0'].isin(list(time_table.iloc[index].time_0))]
        test=test_raw.iloc[:,1:75]
        #print("test:",len(test))
        test.columns=column_names

        #train and predict
        params = {'n_estimators':50, 'learning_rate': 0.1,  'max_depth': 10, 'seed': 0,'min_child_weight':3, 
                    'gamma':0, 'colsample_bytree': 0.9,'subsample': 0.9}
        model = xgb.XGBRegressor(objective='reg:gamma', **params)
        model.fit(X, y, eval_metric="mae") 
        ans = model.predict(test)
        print("ans:",len(ans))
        
        tti=pd.DataFrame(ans,columns=['TTI'])
        batch_time.index=range(0,len(batch_time))
        generated=pd.concat([batch_time,tti],axis=1)
        generated.columns=["time_"+str(i-1),"TTI_"+str(i-1)]
        
        print(i,len(df_lst[i-1]))
        
        #store result
        result_lst[b].append(generated)

#output to file
lst=[]
for i in range(0,len(reader3)):
    id_=id_lst.index(reader3["id_road"][i])
    #print(id_)
    df=result_lst[reader3.index[i]%3][id_]
    tti=df[df["time_"+str(id_)]==reader3["time"][i]]["TTI_"+str(id_)].values[0]
    lst.append(tti)
c=pd.DataFrame(lst,columns=['TTI'])
outcome=reader3.join(c)[['id_sample','TTI']]
outcome.to_csv("outcome3.csv",index=False)