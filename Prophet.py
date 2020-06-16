import pandas as pd
from fbprophet import Prophet

id_lst=[276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]

def read_train():
    reader1 = pd.read_csv('D:/GitHub/sodic/train/train_TTI.csv')
    return reader1

def process_train(reader):
    df_tti_lst=[]
    for i in range(0,12):
        df=reader[reader.id_road==id_lst[i]][['time','TTI']]
        df['time']=pd.to_datetime(df['time'])
        df_tti_lst.append(df[['time','TTI']])
    return df_tti_lst

reader=read_train()
df_speed_lst, df_tti_lst=process_train(reader)
train=df_tti_lst[0][df_tti_lst[0].ds<'2019-12-21 00:00:00']
train=train[train.ds>='2019-12-01 00:00:00']

m = Prophet(growth="linear",yearly_seasonality=False,daily_seasonality=False)
m.fit(train)
future = m.make_future_dataframe(periods=300,freq='10min',include_history=False)
forecast = m.predict(future)