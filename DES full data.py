# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:33:53 2023

@author: coolm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


info = pd.read_csv("Des data/sncand_db.csv")
print(info)



d = {}
for i in range(1,6):
    d["y{0}".format(i)] = pd.read_csv("Year {0}.txt".format(i),sep=' ', skipinitialspace = True, skip_blank_lines=True,
                       on_bad_lines="skip", comment = '#', header=None)
    d["y{0}".format(i)] = d["y{0}".format(i)].to_numpy().flatten()
    d["y{0}".format(i)] = pd.DataFrame(d["y{0}".format(i)]).dropna()

data_array = np.array([])

for i in range(len(info)):
    object_type = info.iloc[i,5]
    snid = info.iloc[i,1]
    snid = "des_real_0" + str(snid) + ".dat"
    Fakeid = info.iloc[i,7]
    print(snid)
    print(str(i*100/(len(info)-1)) + "% Done")
    for key, data in d.items():
        for j in range(len(data)):
            file_name = data.iloc[j,0]
            #print(str(file_name))
            
            if file_name == snid:
                if Fakeid == 0:
                    print("true")
                    data_array = np.append(data_array,snid)
                else:
                    break

Des_data = pd.DataFrame({"filename": data_array})
Des_data.to_csv("DES data.csv")





























