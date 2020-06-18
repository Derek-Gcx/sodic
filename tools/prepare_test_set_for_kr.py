import pandas as pd
import numpy as np
import csv
import os
import sys
import copy

all_id = [276183,276184,275911,275912,276240,276241,276264,276265,276268,276269,276737,276738]

def generate_test_set():
    for road_id in all_id:
        df = pd.read_csv("./test/merge/"+str(road_id)+".csv")