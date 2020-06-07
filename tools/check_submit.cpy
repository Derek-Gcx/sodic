import os
import re

pwd = os.getcwd() + r'\tools'# 替换成当前目录
kill = 0
namelist = []
for filename in os.listdir(pwd):
    if(re.search("submit", filename)!=None):    
        kill += 1
        namelist.append(filename)

if(kill > 1):
    for name in namelist:
        os.remove(pwd+"\\"+name)
    
count = 0
for line in open("./train/submit.csv"):
    line = line.split(',')
    if(count == 0):
        assert(line[0] == 'id_sample')
        assert(line[1].replace('\n', '') == 'TTI')
        count += 1
    else:
        count += 1
        assert(0<float(line[1].replace('\n', ''))<50)
assert(count == 3025)