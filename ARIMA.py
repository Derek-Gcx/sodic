from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

id_lst=[276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]

def read_train():
    reader1 = pd.read_csv('D:/GitHub/sodic/train/train_TTI.csv')
    reader2 = pd.read_csv('D:/GitHub/sodic/train/toPredict_train_TTI.csv')
    reader = pd.concat([reader1,reader2],axis=0)
    reader = reader.drop_duplicates()
    return reader

def process_train(reader):
    date_range1 = pd.date_range(start="2019-01-01 00:00:00", end="2019-03-31 23:50:00", freq="10T")
    date_range2 = pd.date_range(start="2019-10-01 00:00:00", end="2019-12-21 23:50:00", freq="10T")
    df_tti_lst=[]
    for i in range(0,12):
        df=reader[reader.id_road==id_lst[i]][['time','TTI']]
        df = df.set_index("time")
        df = df.set_index(pd.to_datetime(df.index))
        new_df1 = df.reindex(date_range1, fill_value=None)
        new_df2 = df.reindex(date_range2, fill_value=None)
        new_df = new_df1.append(new_df2)
        new_df.fillna(method='ffill',inplace=True)
        df_tti_lst.append(new_df)
    return df_tti_lst

def evaluate():
    sum=0
    toPredict_whole=pd.read_csv('D:/GitHub/sodic/train/toPredict_noLabel.csv')
    order_lst=[(0,2,0)]*12
    order_lst[2]=(0,1,2)
    order_lst[3]=(2,1,2)
    order_lst[4]=(1,0,0)
    order_lst[10]=(0,1,1)
    order_lst[11]=(1,1,2)
    for i in range(0,12):
        raw=df_tti_lst[i]
        toPredict_whole['time']=pd.to_datetime(toPredict_whole['time'])
        toPredict=toPredict_whole[toPredict_whole.id_road==id_lst[i]]
        toPredict_simple=toPredict[toPredict.time<'2019-12-22 00:00:00']
        result=pd.DataFrame(columns=['TTI'])
        train=raw[raw.index<toPredict_simple[0:1].time.values[0]]
        train=train[train.index>'2019-10-01 00:00:00']
        for j in range(0,len(toPredict_simple)):
            model = ARIMA(train.TTI, order=order_lst[i])
            model_fit=model.fit(disp=0)
            pre=model_fit.forecast()

            new1=pd.DataFrame({'TTI':[pre[0][0]]},index=[toPredict_simple[j:j+1].time.values[0]])
            #print(new1)
            result=result.append(new1)
            train=train.append(new1)

            new2=raw[raw.index>toPredict_simple[j:j+1].time.values[0]]
            if j!=len(toPredict_simple)-1:
                new2=new2[new2.index<toPredict_simple[j+1:j+2].time.values[0]]
            #print(new2)
            train=pd.concat([train,new2],axis=0)
            
        compare=pd.merge(result,raw,left_index=True,right_on=raw.index)
        #print(compare)
        x1=compare.index
        y1=compare.TTI_x
        y2=compare.TTI_y
        plt.ylim(0,6)
        plt.plot(x1,y1)
        plt.plot(x1,y2)
        plt.show()
        loss=(abs(compare.TTI_x-compare.TTI_y).sum())/len(compare)
        print('MAE=%.3f' % loss)
        sum+=loss
    print('average MAE=%.3f' % (sum/12))


reader=read_train()
df_tti_lst=process_train(reader)

result_whole=[]
toPredict_whole=pd.read_csv('D:/GitHub/sodic/train/toPredict_noLabel.csv')
order_lst=[(0,2,0)]*12
order_lst[2]=(1,1,2)
order_lst[3]=(1,1,2)
order_lst[4]=(1,0,0)
order_lst[10]=(1,1,1)
order_lst[11]=(1,1,2)
for i in range(0,12):
    raw=df_tti_lst[i]
    toPredict_whole['time']=pd.to_datetime(toPredict_whole['time'])
    toPredict=toPredict_whole[toPredict_whole.id_road==id_lst[i]]
    result=pd.DataFrame(columns=['TTI'])
    train=raw[raw.index<toPredict[0:1].time.values[0]]
    for j in range(0,len(toPredict)):
        model = ARIMA(train.TTI, order=order_lst[i])
        model_fit=model.fit(disp=0)
        pre=model_fit.forecast()

        new1=pd.DataFrame({'TTI':[pre[0][0]]},index=[toPredict[j:j+1].time.values[0]])
        #print(new1)
        result=result.append(new1)
        train=train.append(new1)

        new2=raw[raw.index>toPredict[j:j+1].time.values[0]]
        if j!=len(toPredict)-1:
            new2=new2[new2.index<toPredict[j+1:j+2].time.values[0]]
        #print(new2)
        train=pd.concat([train,new2],axis=0)
        #print(len(train))
    result_whole.append(result)
    print("finish：",i)

for i in range(0,12):
    path='D:/GitHub/sodic/outcome/'+str(i)+'.csv'
    result_whole[i].to_csv(path)