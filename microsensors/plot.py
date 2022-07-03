from importlib.resources import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path='microsensors/data/dump_nilu_micro_sensors.csv'

df = pd.read_csv(path)

fig, axs = plt.subplots(figsize=(655, 7))
plt.plot(df['time'].to_numpy(), df['torget_pm10_nilu'].to_numpy(),label='NILU')
plt.plot(df['time'].to_numpy(), df['torget_pm10'].to_numpy(),label='airRohr')
plt.legend()
#plt.title(title,fontsize=30)
plt.xticks(fontsize=1,rotation = 90)
plt.savefig('torget_pm10.pdf')

df = pd.read_csv(path)

fig, axs = plt.subplots(figsize=(655, 7))
plt.plot(df['time'].to_numpy(), df['torget_pm25_nilu'].to_numpy(),label='NILU')
plt.plot(df['time'].to_numpy(), df['torget_pm25'].to_numpy(),label='airRohr')
plt.legend()
#plt.title(title,fontsize=30)
plt.xticks(fontsize=1,rotation = 90)
plt.savefig('torget_pm25.pdf')

df = pd.read_csv(path)

fig, axs = plt.subplots(figsize=(655, 7))
plt.plot(df['time'].to_numpy(), df['elgeseter_pm10_nilu'].to_numpy(),label='NILU')
plt.plot(df['time'].to_numpy(), df['elgeseter_pm10'].to_numpy(),label='airRohr')
plt.legend()
#plt.title(title,fontsize=30)
plt.xticks(fontsize=1,rotation = 90)
plt.savefig('elgeseter_pm10.pdf')

df = pd.read_csv(path)

fig, axs = plt.subplots(figsize=(655, 7))
plt.plot(df['time'].to_numpy(), df['elgeseter_pm25_nilu'].to_numpy(),label='NILU')
plt.plot(df['time'].to_numpy(), df['elgeseter_pm25'].to_numpy(),label='airRohr')
plt.legend()
#plt.title(title,fontsize=30)
plt.xticks(fontsize=1,rotation = 90)
plt.savefig('elgeseter_pm25.pdf')