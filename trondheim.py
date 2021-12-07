# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
trond_path = "data/dump_with_all_features.csv"
trond = pd.read_csv(trond_path, sep=',')
wx_gk['CO_x']=trond['elgeseter_pm25_nilu']