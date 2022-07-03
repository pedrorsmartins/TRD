'''

Example of using a LSTM encoder-decoder to model a synthetic time series 

'''
#from keras.preprocessing.sequence import TimeseriesGenerator
from importlib.resources import path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import reload
import sys
import pandas as pd

import generate_dataset
import lstm_encoder_decoder
import plotting 

matplotlib.rcParams.update({'font.size': 17})

#----------------------------------------------------------------------------------------------------------------
# generate dataset for LSTM

path_to_dataset_pm10 = ['airRohr/data/pm10/dump_esp8266-13543340.csv','airRohr/data/pm10/dump_esp8266-244085-13543340.csv',
'airRohr/data/pm10/dump_esp8266-244085-240636.csv','airRohr/data/pm10/dump_esp8266-244085.csv','airRohr/data/pm10/dump_esp8266-240636.csv']

titles_pm10 = ['pm10 data acquired from 2022-03-11 until 2022-04-26 (airRohr #3)','pm10 data acquired from 2021-12-17 until 2022-01-13 and from 2022-03-11 until 2022-04-26 (airRohr #1 and #3)',
'pm10 data acquired from 2021-12-17 until 2022-01-13 and from 2022-03-11 until 2022-04-10 (airRohr #1 and #2)','pm10 data acquired from 2021-12-17 until 2022-01-13 (airRoh #1)','pm10 data acquired from 2022-03-11 until 2022-04-10 (airRohr #2)']

path_to_dataset_pm25 = ['airRohr/data/pm25/dump_esp8266-13543340.csv','airRohr/data/pm25/dump_esp8266-244085-13543340.csv',
'airRohr/data/pm25/dump_esp8266-244085-240636.csv','airRohr/data/pm25/dump_esp8266-244085.csv','airRohr/data/pm25/dump_esp8266-240636.csv']

titles_pm25 = ['pm2.5 data acquired from 2022-03-11 until 2022-04-26 (airRohr #3)','pm2.5 data acquired from 2021-12-17 until 2022-01-13 and from 2022-03-11 until 2022-04-26 (airRohr #1 and #3)',
'pm2.5 data acquired from 2021-12-17 until 2022-01-13 and from 2022-03-11 until 2022-04-10 (airRohr #1 and #2)','pm2.5 data acquired from 2021-12-17 until 2022-01-13 (airRoh #1)','pm2.5 data acquired from 2022-03-11 until 2022-04-10 (airRohr #2)']

dataset = 0 #from 0 to 4

pollutant = 'pm25' #'pm10' or 'pm25'

#df = pd.read_csv(path_to_dataset_pm10[dataset])
#df = pd.read_csv(path_to_dataset_pm25[dataset])

df = pd.read_csv('microsensors/data/dump_nilu_micro_sensors.csv')

df = df.dropna()
#t = np.array(range(1,df['Time'].size+1))
t = np.array(range(1,df['time'].size+1))
#y = df['torget_' + pollutant + '_iot'].to_numpy()
y = df['torget_' + pollutant].to_numpy()
y_1 = df['torget_' + pollutant + '_nilu'].to_numpy()
y = array=np.transpose(np.vstack((y,y_1)))

t_train, y_train, t_test, y_test = generate_dataset.train_test_split(t, y, split = 0.8)

# plot time series 
plt.figure(figsize = (28, 6))
plt.plot(t, y[:,0], color = 'k', linewidth = 2, label = 'AirRohr')
plt.plot(t, y[:,1], color = 'b', linewidth = 1.3, label = 'NILU')
plt.xlim([t[0], t[-1]])
plt.xlabel('$t$')
plt.ylabel(pollutant)
plt.legend(bbox_to_anchor=(1, 1))
plt.title('Time Series')
plt.savefig('time_series.png')

# plot time series with train/test split
plt.figure(figsize = (28, 6))
plt.plot(t_train, y_train[:,0], color = '0.2', linewidth = 2, label = 'Train') 
plt.plot(t_train, y_train[:,1], color = '0.6', linewidth = 1.3, label = 'Train') 
plt.plot(np.concatenate([[t_train[-1]], t_test]), np.concatenate([[y_train[-1]], y_test])[:,0],
         color = (0.51, 0.17, 0.03), linewidth = 2, label = 'Test')
plt.plot(np.concatenate([[t_train[-1]], t_test]), np.concatenate([[y_train[-1]], y_test])[:,1],
        color = (0.86, 0.47, 0.31), linewidth = 1.3, label = 'Test')
plt.xlim([t[0], t[-1]])
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.title('Time Series Split into Train and Test Sets')
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout
plt.savefig('train_test_split.png')

#----------------------------------------------------------------------------------------------------------------
# window dataset

# set size of input/output windows 
iw = 20 
ow = 1
s = 1

# generate windowed training/test datasets



Xtrain, Ytrain= generate_dataset.windowed_dataset(y_train, input_window = iw, output_window = ow, stride = s, num_features=1)
Xtest, Ytest = generate_dataset.windowed_dataset(y_test, input_window = iw, output_window = ow, stride = s, num_features=1)

# plot example of windowed data  
plt.figure(figsize = (10, 6)) 
plt.plot(np.arange(0, iw), Xtrain[:, 0,0], 'k', linewidth = 2.2, label = 'Input')
plt.plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, 0,0]], Ytrain[:, 0]]),
         color = (0.2, 0.42, 0.72), linewidth = 2.2, label = 'Target')
plt.xlim([0, iw + ow - 1])
plt.xlabel(r'$t$')
plt.ylabel(r'$y$')
plt.title('Example of Windowed Training Data')
plt.legend(bbox_to_anchor=(1.3, 1))
plt.tight_layout() 
plt.savefig('windowed_data.png')

#----------------------------------------------------------------------------------------------------------------
# LSTM encoder-decoder

# convert windowed data from np.array to PyTorch tensor
X_train, Y_train, X_test, Y_test = generate_dataset.numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)

# specify model parameters and train
model = lstm_encoder_decoder.lstm_seq2seq(input_size = X_train.shape[2], hidden_size = 15)
loss = model.train_model(X_train, Y_train, n_epochs = 50, target_len = ow, batch_size = 5, training_prediction = 'mixed_teacher_forcing', teacher_forcing_ratio = 0.6, learning_rate = 0.01, dynamic_tf = False)

# plot predictions on train/test data
#plotting.plot_train_test_results(model, Xtrain, Ytrain, Xtest, Ytest, titles_pm10[dataset])
#plotting.plot_train_test_results(model, Xtrain, Ytrain, Xtest, Ytest, titles_pm25[dataset])

plotting.plot_train_test_results(model, Xtrain, Ytrain, Xtest, Ytest, 'torget_pm25')

plt.close('all')

