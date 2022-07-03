# Author: Laura Kulowski

import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_train_test_results(lstm_model, Xtrain, Ytrain, Xtest, Ytest, title, num_rows = 4):
  '''
  plot examples of the lstm encoder-decoder evaluated on the training/test data
  
  : param lstm_model:     trained lstm encoder-decoder
  : param Xtrain:         np.array of windowed training input data
  : param Ytrain:         np.array of windowed training target data
  : param Xtest:          np.array of windowed test input data
  : param Ytest:          np.array of windowed test target data 
  : param num_rows:       number of training/test examples to plot
  : return:               num_rows x 2 plots; first column is training data predictions,
  :                       second column is test data predictions
  '''

  # input window size
  iw = Xtrain.shape[0]
  ow = Ytest.shape[0]

  # figure setup 
  num_cols = 2
  num_plots = num_rows * num_cols

  fig, ax = plt.subplots(num_rows, num_cols, figsize = (13, 15))

  print(Xtrain.shape)
  print(Ytrain.shape)
  print(Ytest.shape)

  Y_train_pred = lstm_model.predict(torch.from_numpy(Xtrain).type(torch.Tensor), target_len = ow)
  Y_test_pred = lstm_model.predict(torch.from_numpy(Xtest).type(torch.Tensor), target_len = ow)

  print(Y_train_pred.shape)
  Y_train_pred = np.squeeze(Y_train_pred)
  Y_test_pred = np.squeeze(Y_test_pred)
  Ytest = np.squeeze(Ytest)
  Ytrain = np.squeeze(Ytrain)

  fig, axs = plt.subplots(figsize=(30, 7))
  plt.plot(np.arange(len(Y_train_pred)), Y_train_pred, label='Prediction')
  plt.plot(np.arange(len(Ytrain)), Ytrain, label='Training data')
  plt.legend()
  plt.title(title,fontsize=11)
  #plt.xticks(fontsize=1,rotation = 90)
  plt.savefig('predictions_train.png')

  fig, axs = plt.subplots(figsize=(30, 7))
  plt.plot(np.arange(len(Y_test_pred)), Y_test_pred,label='Prediction')
  plt.plot(np.arange(len(Ytest)), Ytest, label='Testing data')
  plt.legend()
  plt.title(title,fontsize=11)
  #plt.xticks(fontsize=1,rotation = 90)
  plt.savefig('predictions_test.png')

  # plot training/test predictions
  # for ii in range(num_rows):
  #     # train set
  #     X_train_plt = Xtrain[:, ii, :]
  #     Y_train_pred = lstm_model.predict(torch.from_numpy(X_train_plt).type(torch.Tensor), target_len = ow)
  #     print(Y_train_pred.shape)
  #     ax[ii, 0].plot(np.arange(0, iw), Xtrain[:, ii, 0], 'k', linewidth = 2, label = 'Input')
  #     ax[ii, 0].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, ii, 0]], Ytrain[:, ii]]),
  #                    color = (0.2, 0.42, 0.72), linewidth = 2, label = 'Target')
  #     ax[ii, 0].plot(np.arange(iw - 1, iw + ow),  np.concatenate([[Xtrain[-1, ii, 0]], Y_train_pred[:, 0]]),
  #                    color = (0.76, 0.01, 0.01), linewidth = 2, label = 'Prediction')
  #     ax[ii, 0].set_xlim([0, iw + ow - 1])
  #     ax[ii, 0].set_xlabel('$t$')
  #     ax[ii, 0].set_ylabel('$y$')

  #     # test set
  #     X_test_plt = Xtest[:, ii, :]
  #     Y_test_pred = lstm_model.predict(torch.from_numpy(X_test_plt).type(torch.Tensor), target_len = ow)
  #     ax[ii, 1].plot(np.arange(0, iw), Xtest[:, ii, 0], 'k', linewidth = 2, label = 'Input')
  #     ax[ii, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, ii, 0]], Ytest[:, ii]]),
  #                    color = (0.2, 0.42, 0.72), linewidth = 2, label = 'Target')
  #     ax[ii, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtest[-1, ii, 0]], Y_test_pred[:, 0]]),
  #                    color = (0.76, 0.01, 0.01), linewidth = 2, label = 'Prediction')
  #     ax[ii, 1].set_xlim([0, iw + ow - 1])
  #     ax[ii, 1].set_xlabel('$t$')
  #     ax[ii, 1].set_ylabel('$y$')

  #     if ii == 0:
  #       ax[ii, 0].set_title('Train')
        
  #       ax[ii, 1].legend(bbox_to_anchor=(1, 1))
  #       ax[ii, 1].set_title('Test')

  # plt.suptitle('LSTM Encoder-Decoder Predictions', x = 0.445, y = 1.)
  # plt.tight_layout()
  # plt.subplots_adjust(top = 0.95)
  # plt.savefig('predictions.png')
  # plt.close() 
      
  return 
