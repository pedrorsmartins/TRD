# encoding=gbk
import os
from train_code.common import *
from train_code.train import train
import configparser
import numpy as np


if __name__ == '__main__':
    wx_path = 'data/example3.csv' #location of dataset( Note that we have no right to directly publish the datasets in our paper.)
    wx_gk = pd.read_csv(wx_path, sep=';')
    wx_title = wx_path.split('/')[-1].split('.')[0]
    model_path = 'data/model'
    check_or_create_path(model_path)
    trond_path = 'data/dump_with_all_features.csv'
    trond = pd.read_csv(trond_path, sep=',')
    # wx_gk['CO_x']=wx_gk['NO2_x']=wx_gk['SO2_x']=wx_gk['O3_x']=wx_gk['PM10_x']=0.0
    # wx_gk['PM25_x']=trond['elgeseter_pm25_nilu']
    wx_gk['CO_x']=wx_gk['NO2_x']=wx_gk['SO2_x']=wx_gk['O3_x']=wx_gk['PM10_x']=wx_gk['PM25_x']=trond['elgeseter_pm25_nilu']
    # wx_gk['CO_y']=wx_gk['NO2_y']=wx_gk['SO2_y']=wx_gk['O3_y']=wx_gk['PM10_y']=0
    # wx_gk['PM25_y']=trond['elgeseter_pm25_iot']
    wx_gk['CO_y']=wx_gk['NO2_y']=wx_gk['SO2_y']=wx_gk['O3_y']=wx_gk['PM10_y']=wx_gk['PM25_x']=trond['elgeseter_pm25_iot']
    wx_gk['RECEIVETIME']=trond['time']
    wx_gk['TEMPERATURE']=trond['air_temperature']
    wx_gk['HUMIDITY']=trond['relative_humidity']
    train(wx_gk, wx_title, model_path)
    