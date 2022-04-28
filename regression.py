import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

from sklearn.metrics import mean_squared_error

use_temporal_features = True
use_opc_weather = False
use_opc_weather = False
use_weather = False
use_only_temp_hum = True
use_traffic = False
use_all_pm = False
use_no_pm = False

split_by_time = True
split_random = not split_by_time

train_percentage = 0.75

train_sensor = 'torget'
test_sensor = 'torget'
pollutant = 'pm25'

path_to_dataset = 'data/dump_esp8266-244085-13543340.csv'
#path_to_dataset = 'data/dump_esp8266-244085-240636.csv'
#path_to_dataset = 'data/dump_esp8266-244085.csv'
#path_to_dataset = 'data/dump_esp8266-240636.csv'
#path_to_dataset = 'data/dump_esp8266-13543340.csv'

save_model = False

df = pd.read_csv(path_to_dataset)


# df = df.set_index('Time')

# print(df.head())
# print(df.columns)

if use_temporal_features == True:
    import datetime
    df = df.set_index(pd.to_datetime(df['Time'],utc=True))
    df['hour_of_day'] = df.index.hour + 1
    #df['month_of_year'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    #df['day_of_month'] = df.index.day
    #df['day_of_year'] = df.index.dayofyear
    #df['season'] = (df.index.month%12 + 3)//3
    df.reset_index(drop=True,inplace=True)
    df = df.drop(columns='Time')
    

df = df.dropna()

# if use_weather == False:
#     df = df.drop(['air_temperature','wind_speed','precipitation','relative_humidity','air_pressure','wind_direction'], axis=1)

# if use_weather == True and use_only_temp_hum == True:
#     df = df.drop(['wind_speed','precipitation','air_pressure','wind_direction'], axis=1)
    
# if use_opc_weather == False:
#     df = df.drop(['elgeseter_opctemp_iot','elgeseter_opchum_iot','torget_opctemp_iot','torget_opchum_iot'], axis=1)
    
# if use_traffic == False:
#     df = df.drop(['16219V72812_Total','44656V72812_Total','10236V72161_Total'],axis=1)
    
# #Remove columns with data from other sensors
# df = df.drop(df.columns[~df.columns.str.startswith(train_sensor) & df.columns.str.endswith('iot')],axis=1)

# if use_all_pm == False:
#     df = df.drop(df.columns[df.columns.str.startswith(train_sensor) & df.columns.str.contains('pm') & ~df.columns.str.contains(pollutant) & df.columns.str.endswith('iot')],axis=1)
    
# if use_no_pm == True:
#     df = df.drop(df.columns[df.columns.str.endswith('iot')],axis=1)

# #Remove nilu data which is not the target
# #1) Remove data from other sensors; 2) remove data from the target sensor which is not the target pollutant
# df = df.drop(df.columns[~df.columns.str.startswith(train_sensor) & df.columns.str.endswith('nilu')],axis=1)
# df = df.drop(df.columns[df.columns.str.startswith(train_sensor) & ~df.columns.str.contains(pollutant) & df.columns.str.endswith('nilu')],axis=1)

# df = df.drop(['time'],axis=1)

# df = df.dropna()

# Set random seed to ensure reproducible runs
RSEED = 50

# Extract the labels
labels = np.array(df.pop(train_sensor+'_'+pollutant+'_nilu'))

if split_by_time == True:
    train_labels = labels[:int(labels.shape[0]*train_percentage)]
    test_labels = labels[int(labels.shape[0]*train_percentage):]

    train = df.iloc[:int(df.shape[0]*train_percentage)]
    test = df.iloc[int(df.shape[0]*train_percentage):]


elif split_random == True:
    from sklearn.model_selection import train_test_split

    train, test, train_labels, test_labels = train_test_split(df, labels, 
                                                          test_size = (1-train_percentage), 
                                                          random_state = RSEED)

# train = train.dropna()
# test = test.dropna()

# Features for feature importances
features = list(train.columns)

train.shape

test.shape

train.isnull().sum()


from sklearn.ensemble import RandomForestRegressor

if int(len(features)/3) > 0:
    mf = int(len(features)/3)
else:
    mf = 1

# Create the model with 100 trees
model = RandomForestRegressor(n_estimators=100,
                               max_depth = 10,
                               random_state=RSEED, 
                               max_features = mf,
                               n_jobs=-1, verbose = 1)

# Fit on training data
model.fit(train, train_labels)


# We can see how many nodes there are for each tree on average and the maximum depth of each tree. 
# There were 100 trees in the forest.
n_nodes = []
max_depths = []

for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

train_rf_predictions = model.predict(train)

test_rf_predictions = model.predict(test)

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

def plot_predictions(labels,predictions):
    fontsize = 20
    fig_x_size = 17.14
    fig_y_size = 10
    marker_size = 10

    x=labels
    y=predictions
    x = x.reshape(len(x), 1)
    y = y.reshape(len(y), 1)

    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    plt.figure(figsize=(fig_x_size,fig_y_size))
    plt.scatter(x, y,  color='black',s=marker_size)
    plt.plot(x, regr.predict(x), color='blue', linewidth=3)
    plt.ylabel(r'Predicted concentration ($\mu$g/m$^3$)',fontsize=fontsize)
    plt.xlabel(r'Observed concentration ($\mu$g/m$^3$)',fontsize=fontsize)
    plt.yticks(fontsize=fontsize) 
    plt.xticks(fontsize=fontsize)   
    #plt.xticks(())
    #plt.yticks(())
    #plt.text(60, 10, 'R-squared = %0.2f' % r2_score(x,y))
    #plt.show()
    plt.savefig('fig.pdf')
    plt.close()
    
    print('Coefficients: {}x + {}'.format(regr.coef_,regr.intercept_))

plot_predictions(test_labels,test_rf_predictions)

fi_model = pd.DataFrame({'feature': features,
                'importance': model.feature_importances_}).\
                sort_values('importance', ascending = False)
fi_model.head(40)

#fi_model.to_dict()

df = pd.read_csv(path_to_dataset)

#df[shift_cols] = df[shift_cols].shift(shift_time)

## Starts here

if use_temporal_features == True:
    import datetime
    df = df.set_index(pd.to_datetime(df['Time'],utc=True))
    df['hour_of_day'] = df.index.hour+1
    #df['month_of_year'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    #df['day_of_month'] = df.index.day
    #df['day_of_year'] = df.index.dayofyear
    #df['season'] = (df.index.month%12 + 3)//3
    df.reset_index(drop=True,inplace=True)
  
    
# if use_weather == False:
#     df = df.drop(['air_temperature','wind_speed','precipitation','relative_humidity','air_pressure','wind_direction'], axis=1)

# if use_weather == True and use_only_temp_hum == True:
#     df = df.drop(['wind_speed','precipitation','air_pressure','wind_direction'], axis=1)
    
# if use_opc_weather == False:
#     df = df.drop(['elgeseter_opctemp_iot','elgeseter_opchum_iot','torget_opctemp_iot','torget_opchum_iot'], axis=1)
    
# if use_traffic == False:
#     df = df.drop(['16219V72812_Total','44656V72812_Total','10236V72161_Total'],axis=1)
    
# #Remove columns with data from other sensors
# df = df.drop(df.columns[~df.columns.str.startswith(test_sensor) & df.columns.str.endswith('iot')],axis=1)

# if use_all_pm == False:
#     df = df.drop(df.columns[df.columns.str.startswith(test_sensor) & df.columns.str.contains('pm') & ~df.columns.str.contains(pollutant) & df.columns.str.endswith('iot')],axis=1)
    
# if use_no_pm == True:
#     df = df.drop(df.columns[df.columns.str.endswith('iot')],axis=1)

# #Remove nilu data which is not the target
# #1) Remove data from other sensors; 2) remove data from the target sensor which is not the target pollutant
# df = df.drop(df.columns[~df.columns.str.startswith(test_sensor) & df.columns.str.endswith('nilu')],axis=1)
# df = df.drop(df.columns[df.columns.str.startswith(test_sensor) & ~df.columns.str.contains(pollutant) & df.columns.str.endswith('nilu')],axis=1)

df = df.drop(['Time'],axis=1)

df = df.dropna()

test_sensor_labels = np.array(df.pop(test_sensor+'_'+pollutant+'_nilu'))


test_sensor_rf_predictions = model.predict(df)

plot_predictions(test_sensor_labels,test_sensor_rf_predictions)

with open('results.txt', 'w') as f:
    f.write('Train '+train_sensor+' RMSE: %.2f' 
      % mean_squared_error(train_rf_predictions,train_labels,squared=False))
    f.write('\n')
    f.write('Train '+train_sensor+' R^2: %.2f'
      % r2_score(train_labels,train_rf_predictions))
    f.write('\n')
    f.write('Test '+train_sensor+' RMSE: %.2f' 
      % mean_squared_error(test_rf_predictions,test_labels,squared=False))
    f.write('\n')
    f.write('Test '+train_sensor+' R^2: %.2f'
      % r2_score(test_labels,test_rf_predictions))
    f.write('\n')
    f.write('Test '+test_sensor+' RMSE: %.2f' 
      % mean_squared_error(test_sensor_rf_predictions,test_sensor_labels,squared=False))
    f.write('\n')
    f.write('Test '+test_sensor+' sensor R^2: %.2f'
      % r2_score(test_sensor_labels,test_sensor_rf_predictions))
    f.write('\n')

print('Train '+train_sensor+' RMSE: %.2f' 
      % mean_squared_error(train_rf_predictions,train_labels,squared=False))
print('Train '+train_sensor+' R^2: %.2f'
      % r2_score(train_labels,train_rf_predictions))

print('\n')

print('Test '+train_sensor+' RMSE: %.2f' 
      % mean_squared_error(test_rf_predictions,test_labels,squared=False))
print('Test '+train_sensor+' R^2: %.2f'
      % r2_score(test_labels,test_rf_predictions))

print('\n')


print('Test '+test_sensor+' RMSE: %.2f' 
      % mean_squared_error(test_sensor_rf_predictions,test_sensor_labels,squared=False))
print('Test '+test_sensor+' sensor R^2: %.2f'
      % r2_score(test_sensor_labels,test_sensor_rf_predictions))


fig, axs = plt.subplots(figsize=(100, 7))
plt.plot(train_labels,label='NILU')
plt.plot(train_rf_predictions,label='Train predictions')
plt.legend()
plt.savefig('fig3.pdf')

fig, axs = plt.subplots(figsize=(100, 7))
plt.plot(test_labels,label='NILU')
plt.plot(test_rf_predictions,label='Test predictions')
plt.legend()
plt.savefig('fig4.pdf')

fig, axs = plt.subplots(figsize=(100, 7))
plt.plot(test_sensor_labels,label='NILU')
plt.plot(test_sensor_rf_predictions,label='Test predictions')
plt.legend()
plt.savefig('fig5.pdf')

save_filepath = path_to_dataset[:-4] + '_target_' + train_sensor + '_' + pollutant + '.pkl'

if save_model == True:
    import joblib
    joblib.dump(model,save_filepath)