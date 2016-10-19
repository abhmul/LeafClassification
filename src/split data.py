import os
import pandas as pd
import numpy as np

data = pd.read_csv('../train.csv')
parent_data = data.copy()    ## Always a good idea to keep a copy of original data
ID = set([int(i) for i in list(data.pop('id'))])

rootdir = '../images/trai'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if int(file[:-4]) in ID:
            try:
                os.rename(rootdir + '/' + file, rootdir + "/train/"+file)
            except:
                print rootdir + '/' + file
        else:
            try:
                os.rename(rootdir + '/' + file, rootdir + "/test/" + file)
            except:
                print rootdir + '/' + file


