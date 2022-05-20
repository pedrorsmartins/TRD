import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from random import randint, choice

from sklearn.metrics import mean_squared_error

use_temporal_features = False
use_opc_weather = False
use_weather = False
use_only_temp_hum = False
use_traffic = False
use_all_pm = False

split_by_time = True
split_random = not split_by_time

train_percentage = 0.75

train_sensor = 'torget'
test_sensor = 'torget'
pollutant = 'pm25'

path_to_dataset = 'data/esp8266-244085.csv'

threshold = 30

df = pd.read_csv(path_to_dataset)

df.head()

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
    
# if use_weather == False:
#     df = df.drop(['air_temperature','wind_speed','precipitation','relative_humidity','air_pressure','wind_direction'], axis=1)

# if use_weather == True and use_only_temp_hum == True:
#     df = df.drop(['wind_speed','precipitation','air_pressure','wind_direction'], axis=1)
    
# if use_opc_weather == False:
#     df = df.drop(['elgeseter_DHT22 temperature_iot','elgeseter_DHT22 humidity_iot'], axis=1)
    
# if use_traffic == False:
#     df = df.drop(['16219V72812_Total','44656V72812_Total','10236V72161_Total'],axis=1)
    
#Remove columns with data from other sensors
df = df.drop(df.columns[~df.columns.str.startswith(train_sensor) & df.columns.str.endswith('iot')],axis=1)

if use_all_pm == False:
    df = df.drop(df.columns[df.columns.str.startswith(train_sensor) & df.columns.str.contains('pm') & ~df.columns.str.contains(pollutant) & df.columns.str.endswith('iot')],axis=1)

#Remove nilu data which is not the target
#1) Remove data from other sensors; 2) remove data from the target sensor which is not the target pollutant
df = df.drop(df.columns[~df.columns.str.startswith(train_sensor) & df.columns.str.endswith('nilu')],axis=1)
df = df.drop(df.columns[df.columns.str.startswith(train_sensor) & ~df.columns.str.contains(pollutant) & df.columns.str.endswith('nilu')],axis=1)

df = df.drop(['Time'],axis=1)

df = df.dropna()

# Create pollution indicator
df["pollution_indicator_max"] = 1.0*(df[train_sensor+"_"+pollutant+"_nilu"] >= threshold)
df['pollution_indicator_max'] = df['pollution_indicator_max'].fillna(0).astype(np.int64)

df['pollution_indicator_max'].value_counts()

import matplotlib as mpl
#matplotlib inline

mpl.rcParams['figure.figsize'] = (20, 10)
mpl.rcParams['axes.grid'] = False

# View the pollution indicator with the measured values for PM10
plot_cols = [train_sensor+'_'+pollutant+'_nilu', 'pollution_indicator_max']
plot_features = df[plot_cols]
_ = plot_features.plot(subplots=True)


# Set random seed to ensure reproducible runs
RSEED = 50

# Extract the labels
labels = np.array(df.pop('pollution_indicator_max'))
#labels = np.array(df.pop(train_sensor+'_'+pollutant+'_nilu'))
df = df.drop([train_sensor+'_'+pollutant+'_nilu'],axis=1)

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

train = train.dropna()
test = test.dropna()

# Features for feature importances
features = list(train.columns)                                                          

train.shape

test.shape

train.isnull().sum()

from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               random_state=RSEED, 
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)

# Fit on training data
model.fit(train, train_labels)

# We can see how many nodes there are for each tree on average and the maximum depth of each tree. 
n_nodes = []
max_depths = []

for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, average_precision_score

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    #baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    #baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['recall'] = recall_score(test_labels, [randint(0, 1) for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [randint(0, 1) for _ in range(len(test_labels))])
    baseline['average_precision'] = average_precision_score(test_labels, [randint(0, 1) for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['average_precision'] = average_precision_score(test_labels, probs)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['average_precision'] = average_precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'average_precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');evaluate_model

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)


train_rf_predictions = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

rf_predictions = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]

fi_model = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi_model.head(40)

df = pd.read_csv(path_to_dataset)

if use_temporal_features == True:
    import datetime
    df = df.set_index(pd.to_datetime(df['time'],utc=True))
    df['hour_of_day'] = df.index.hour + 1
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
#     df = df.drop(['elgeseter_DHT22 temperature_iot','elgeseter_DHT22 humidity_iot'], axis=1)
    
# if use_traffic == False:
#     df = df.drop(['16219V72812_Total','44656V72812_Total','10236V72161_Total'],axis=1)
    
#Remove columns with data from other sensors
df = df.drop(df.columns[~df.columns.str.startswith(test_sensor) & df.columns.str.endswith('iot')],axis=1)

if use_all_pm == False:
    df = df.drop(df.columns[df.columns.str.startswith(test_sensor) & df.columns.str.contains('pm') & ~df.columns.str.contains(pollutant) & df.columns.str.endswith('iot')],axis=1)

#Remove nilu data which is not the target
#1) Remove data from other sensors; 2) remove data from the target sensor which is not the target pollutant
df = df.drop(df.columns[~df.columns.str.startswith(test_sensor) & df.columns.str.endswith('nilu')],axis=1)
df = df.drop(df.columns[df.columns.str.startswith(test_sensor) & ~df.columns.str.contains(pollutant) & df.columns.str.endswith('nilu')],axis=1)

df = df.drop(['Time'],axis=1)

df = df.dropna()

# Create pollution indicator
df["pollution_indicator_max"] = 1.0*(df['torget_pm25_iot'] >= threshold)
df['pollution_indicator_max'] = df['pollution_indicator_max'].fillna(0).astype(np.int64)

df['pollution_indicator_max'].value_counts()

# Extract the labels
test_sensor_labels = np.array(df.pop('pollution_indicator_max'))

df = df.drop(['torget_pm25_iot'],axis=1)

test_sensor_rf_predictions = model.predict(df)
test_sensor_rf_probs = model.predict_proba(df)[:, 1]

def evaluate_model_test_sensor(predictions, probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    #baseline['recall'] = recall_score(test_labels, [1 for _ in range(len(test_labels))])
    #baseline['precision'] = precision_score(test_labels, [1 for _ in range(len(test_labels))])
    baseline['recall'] = recall_score(test_sensor_labels, [randint(0, 1) for _ in range(len(test_sensor_labels))])
    baseline['precision'] = precision_score(test_sensor_labels, [randint(0, 1) for _ in range(len(test_sensor_labels))])
    baseline['average_precision'] = average_precision_score(test_sensor_labels, [randint(0, 1) for _ in range(len(test_sensor_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_sensor_labels, predictions)
    results['precision'] = precision_score(test_sensor_labels, predictions)
    results['average_precision'] = average_precision_score(test_sensor_labels, probs)
    results['roc'] = roc_auc_score(test_sensor_labels, probs)
    
    
    for metric in ['recall', 'precision', 'average_precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_sensor_labels, [1 for _ in range(len(test_sensor_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_sensor_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');

evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)

cm = confusion_matrix(test_labels, rf_predictions)
plot_confusion_matrix(cm, classes = ['Low pollution', 'High pollution'],
                      title = 'Pollution Confusion Matrix '+train_sensor,normalize=False)

evaluate_model_test_sensor(test_sensor_rf_predictions, test_sensor_rf_probs)

cm = confusion_matrix(test_sensor_labels, test_sensor_rf_predictions)
plot_confusion_matrix(cm, classes = ['Low pollution', 'High pollution'],
                      title = 'Pollution Confusion Matrix '+test_sensor,normalize=False)

df.corr(method='pearson')


print('Hello')