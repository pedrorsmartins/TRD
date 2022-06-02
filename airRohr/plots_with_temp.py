import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




path_to_dataset_pm10 = ['airRohr/data/pm10/dump_esp8266-13543340.csv','airRohr/data/pm10/dump_esp8266-244085-13543340.csv',
'airRohr/data/pm10/dump_esp8266-244085-240636.csv','airRohr/data/pm10/dump_esp8266-244085.csv','airRohr/data/pm10/dump_esp8266-240636.csv']

titles_pm10 = ['pm10 data acquired from 2022-03-11 until 2022-04-26 (airRohr #3)','pm10 data acquired from 2021-12-17 until 2022-01-13 and from 2022-03-11 until 2022-04-26 (airRohr #1 and #3)',
'pm10 data acquired from 2021-12-17 until 2022-01-13 and from 2022-03-11 until 2022-04-10 (airRohr #1 and #2)','pm10 data acquired from 2021-12-17 until 2022-01-13 (airRoh #1)','pm10 data acquired from 2022-03-11 until 2022-04-10 (airRohr #2)']

aux=0
  
for title,path in zip(titles_pm10,path_to_dataset_pm10):
  aux+=1
  df = pd.read_csv(path)

  fig, ax1 = plt.subplots(figsize=(100, 7))
  plt.plot(df['Time'].to_numpy(), df['torget_pm10_nilu'].to_numpy(),label='NILU')
  plt.plot(df['Time'].to_numpy(), df['torget_pm10_iot'].to_numpy(),label='airRohr')
  plt.legend()
  plt.title(title,fontsize=30)
  plt.xticks(fontsize=4,rotation = 90)
  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:green'
  ax2.set_ylabel('Temperature (°C)', color=color)  # we already handled the x-label with ax1
  ax2.plot(df['Time'].to_numpy(),df['air_temperature'].to_numpy() , color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  plt.savefig('fig'+str(aux)+'.pdf')

path_to_dataset_pm25 = ['airRohr/data/pm25/dump_esp8266-13543340.csv','airRohr/data/pm25/dump_esp8266-244085-13543340.csv',
'airRohr/data/pm25/dump_esp8266-244085-240636.csv','airRohr/data/pm25/dump_esp8266-244085.csv','airRohr/data/pm25/dump_esp8266-240636.csv']

titles_pm25 = ['pm2.5 data acquired from 2022-03-11 until 2022-04-26 (airRohr #3)','pm2.5 data acquired from 2021-12-17 until 2022-01-13 and from 2022-03-11 until 2022-04-26 (airRohr #1 and #3)',
'pm2.5 data acquired from 2021-12-17 until 2022-01-13 and from 2022-03-11 until 2022-04-10 (airRohr #1 and #2)','pm2.5 data acquired from 2021-12-17 until 2022-01-13 (airRoh #1)','pm2.5 data acquired from 2022-03-11 until 2022-04-10 (airRohr #2)']


aux=5
for title,path in zip(titles_pm25,path_to_dataset_pm25):
  aux+=1
  df = pd.read_csv(path)

  fig, axs = plt.subplots(figsize=(100, 7))
  plt.plot(df['Time'].to_numpy(), df['torget_pm25_nilu'].to_numpy(),label='NILU')
  plt.plot(df['Time'].to_numpy(), df['torget_pm25_iot'].to_numpy(),label='airRohr')
  plt.legend()
  plt.title(title,fontsize=30)
  plt.xticks(fontsize=4,rotation = 90)
  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:green'
  ax2.set_ylabel('Temperature (°C)', color=color)  # we already handled the x-label with ax1
  ax2.plot(df['Time'].to_numpy(),df['air_temperature'].to_numpy() , color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  plt.savefig('fig'+str(aux)+'.pdf')