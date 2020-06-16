import pandas as pd
import csv

for k in range(1,21):
    read_path='D:/GitHub/sodic/data_processed/201912/12'+str(k)+'.csv'
    file=open(read_path)
    reader=csv.reader(file)
    total=pd.DataFrame()
    for line in reader:
        line1=[item[1:-1].replace('(','').replace(')','').split(',') for item in line]
        line2=[list(map(eval,item)) for item in line1]
        df=pd.DataFrame([[int(x[0]),x[1]*3.6,x[2]*6+x[3]//10] for x in line2])
        total=total.append(df,ignore_index=True)
    
    total.columns=['road_id','speed','time_block']
    G=total.groupby(['road_id','time_block']).agg(['min', 'max','mean','std']).reset_index()
    G.columns=['road_id','time_block','min','max','mean','std']
    
    id_lst=[276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]
    for road in id_lst:
        write_path='D:/GitHub/sodic/data_processed_2nd/201912/'+str(road)+'/121.csv'
        target=G[G.road_id==road][['time_block','mean','std','min','max']]
        target.to_csv(write_path,index=False)
    
    print("finish:",k)