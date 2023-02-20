import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import max_error, mean_absolute_error, r2_score

from statsmodels.tsa.seasonal import seasonal_decompose

import mlflow
import mlflow.keras

def find_anomalies(sigma, result):
  resid = result.resid
  resid.dropna(inplace=True)

  mean = sum(resid)/len(resid)
  delta = [(resid[k] - mean)**2 for k in range(len(resid))]
  variance = sum(delta)/len(delta)
  deviation = variance**0.5

  high = mean + sigma * deviation
  low = mean - sigma * deviation

  anomalies = resid.copy()

  for k in range(len(anomalies)):
    if resid[k] <= high and resid[k] >= low:
      anomalies.drop(resid.index[k], inplace=True)
  anomalies.index = pd.to_datetime(anomalies.index)

  return anomalies, resid, high, low

def resid_reduction(df, anomalies, resid, high, low):
  for k in range(len(anomalies)):
    if resid[anomalies.index[k]] < low:
      df.loc[anomalies.index[k]] += low - resid[anomalies.index[k]]
    elif resid[anomalies.index[k]] > high:
      df.loc[anomalies.index[k]] += high - resid[anomalies.index[k]]

def train_model(SOURCE_DATA_POWER,
                anomaly_threshold,
                anomaly_same_val_len,
                input_size,
                output_size,
                step,
                split_size,
                n_neurons,
                drop,
                epochs_num):

  seq_size = input_size + output_size
  
  df = pd.read_csv(SOURCE_DATA_POWER, sep = ';', parse_dates=[0])
  df['Time'] = pd.to_datetime(df['Time'])
  df.index = df['Time']
  df = df.drop([df.columns[0],df.columns[2]], axis=1)

  a = pd.DataFrame()
  anomalies_same_val = pd.DataFrame()

  f = 0
  for i in range(1, len(df)):
    if np.isnan(df.iloc[i,0]):
      anomalies_same_val = pd.concat([anomalies_same_val,df.iloc[[i]]])
      anomalies_same_val.iloc[-1, 0] = 0

    delta = abs(df.iloc[i, 0] - df.iloc[i-1, 0])
    if delta <= anomaly_threshold:
      f = 1
      a = pd.concat([a, df.iloc[[i]]])
      if len(a) == 1:
        a = pd.concat([a, df.iloc[[i-1]]])
    elif delta > anomaly_threshold and f == 1:
      f = 0
      if len(a) >= anomaly_same_val_len:
        anomalies_same_val = pd.concat([anomalies_same_val, a])
      a.drop(a.index, inplace=True)

  max_load = max(list(df.iloc[:,0]))
  min_load = min(list(df.iloc[:,0]))
  df = df.replace(np.nan, 2*min_load-max_load)
  for i in range(len(anomalies_same_val)):
    df.loc[anomalies_same_val.index[i]] = 2*min_load-max_load

  for p in range(0, len(df)//h):
    i = h * p
    j = h * (p + 1)
    result = seasonal_decompose(df.iloc[i:j], model='additive')
    anomalies, resid, high, low = find_anomalies(sigma, result)
    if len(anomalies) > 0:
      resid_reduction(df, anomalies, resid, high, low)

  data = df.copy()
  data['Month'] = data.index.month
  data['Week'] = data.index.isocalendar().week.astype(np.int64)
  data['Day of week'] = data.index.dayofweek

  data_scaled = data.copy()
  scalers = []

  for i in range(len(list(data_scaled.columns))):
    feature = np.reshape(list(data_scaled.iloc[:,i]), (len(data_scaled.iloc[:,i]), 1))
    
    if i in [0]:
      scaler = MinMaxScaler(feature_range=(-1,1)).fit(feature)
    else:
      scaler = MinMaxScaler(feature_range=(0,1)).fit(feature)
    scalers.append(scaler)
    scaled_feature = np.reshape(scaler.transform(feature),len(data_scaled.iloc[:,i])).tolist()
    data_scaled.iloc[:,i] = scaled_feature

  total_size = len(data_scaled) - len(data_scaled)%seq_size
  train_size = int((total_size/input_size)*split_size)*input_size
  test_size = total_size - train_size

  # train = np.array(data_scaled.iloc[:train_size])
  # test = np.array(data_scaled.iloc[train_size:total_size])

  # x_train, y_train, x_test, y_test = np.array([]), np.array([]), np.array([]), np.array([])
  # x_train = np.reshape(x_train, (0,len(scalers)))
  # x_test = np.reshape(x_test, (0,len(scalers)))

  # for i in range(0, train_size-seq_size, step):
  #   x_train = np.append(x_train, train[i:i+input_size], axis=0)
  #   y_train = np.append(y_train, train[i+input_size:i+seq_size,0], axis=0)

  # for i in range(0, test_size-seq_size, step):
  #   x_test = np.append(x_test, test[i:i+input_size], axis=0)
  #   y_test = np.append(y_test, test[i+input_size:i+seq_size,0], axis=0)

  train_test_data = np.array(data_scaled.iloc[:])

  x_train, y_train, x_test, y_test = np.array([]), np.array([]), np.array([]), np.array([])
  x_train = np.reshape(x_train, (0,len(scalers)))
  x_test = np.reshape(x_test, (0,len(scalers)))

  j = 1
  for i in range(0, total_size-seq_size, step):
    if j % 8 == 0:
      x_test = np.append(x_test, train_test_data[i:i+input_size], axis=0)
      y_test = np.append(y_test, train_test_data[i+input_size:i+seq_size,0], axis=0)
    else:
      x_train = np.append(x_train, train_test_data[i:i+input_size], axis=0)
      y_train = np.append(y_train, train_test_data[i+input_size:i+seq_size,0], axis=0)
    j += 1

  x_train = np.reshape(x_train, (int(x_train.shape[0]/input_size), input_size, len(scalers)))
  y_train = np.reshape(y_train, (int(y_train.shape[0]/output_size), output_size, 1))
  x_test = np.reshape(x_test, (int(x_test.shape[0]/input_size), input_size, len(scalers)))
  y_test = np.reshape(y_test, (int(y_test.shape[0]/output_size), output_size, 1))

  tf.keras.backend.clear_session()
  model = Sequential()

  model.add(LSTM(n_neurons, input_shape=(input_size, len(scalers)), return_sequences=True))
  model.add(BatchNormalization())
  model.add(Dropout(drop))

  model.add(LSTM(n_neurons, return_sequences=True))
  model.add(BatchNormalization())
  model.add(Dropout(drop))

  model.add(LSTM(n_neurons))
  model.add(BatchNormalization())
  model.add(Dropout(drop))

  model.add(Dense(64))
  model.add(BatchNormalization())
  model.add(Dropout(drop))

  model.add(Dense(output_size, activation="sigmoid"))

  model.compile(optimizer='adam', loss='mean_absolute_error')
  model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=epochs_num, verbose=2) #batch_size=16

  model.evaluate(x_test, y_test, verbose=2)

  return model, data_scaled, total_size, train_size, test_size, scalers, x_train, y_train, x_test, y_test

def predict_get_metrics(model,
                        data_scaled,
                        scalers,
                        k):
  predictions = pd.DataFrame()
  last_datetime = data_scaled.iloc[train_size+step*k+input_size-1].name
  dates = [last_datetime + pd.Timedelta(hours=i) for i in range(hours_num)]
  predictions.index = dates

  predictions['Data'] = 0
  predictions['Month'] = predictions.index.month
  predictions['Week'] = predictions.index.isocalendar().week.astype(np.int64)
  predictions['Day of week'] = predictions.index.dayofweek

  for i in range(1, len(predictions.columns)):
    p = np.reshape(list(predictions.iloc[:,i]), (len(predictions.iloc[:,i]), 1))
    scaled_feature = np.reshape(scalers[i].transform(p),len(predictions.iloc[:,i])).tolist()
    predictions.iloc[:,i] = scaled_feature

  inp = np.reshape(x_test[k], (1, input_size, len(scalers)))
  pred = model.predict(inp, verbose=0).tolist()[0]

  f1 = int(hours_num/output_size)
  f2 = hours_num%output_size
    
  if len(pred) >= hours_num:
    predictions.iloc[:hours_num,0] = pred[:hours_num]
  elif len(pred) < hours_num:
    for i in range(f1):
      predictions.iloc[output_size*(i):output_size*(i+1),0] = pred
      inp = np.append(inp[0], predictions[output_size*i:-output_size*(f1-i-1)-f2], axis=0)
      inp = np.delete(inp, range(output_size), axis=0)
      if i != f1-1:
        inp = np.reshape(inp, (1, input_size, len(scalers)))
        pred = model.predict(inp, verbose=0).tolist()[0]
    if f2 != 0:
      predictions.iloc[-f2:,0] = pred[:f2]

  d = []
  for i in range(f1+1):
    d += [x_test[k][i][0] for i in range(x_test.shape[1])]
    d += [y_test[k][i][0] for i in range(y_test.shape[1])]
  d = d[:hours_num+input_size]
  p = predictions.iloc[:,0].tolist()

  p = np.reshape(p, (1,len(p)))
  p = scalers[0].inverse_transform(p)[0].tolist()

  d = np.reshape(d, (1,len(d)))
  d = scalers[0].inverse_transform(d)[0].tolist()

  g = [i for i in range(input_size + hours_num)]
  h = [i for i in range(input_size, input_size + hours_num)]

  f = plt.figure()
  f.set_figwidth(30)

  plt.plot(g, d, label='Real')
  plt.plot(h, p, label='Prediction')
  plt.legend()
  plt.savefig(fname = plots_path + plot_name)
  plt.close()
  
  me = max_error(d[input_size:input_size + hours_num], p[:hours_num])
  mae = mean_absolute_error(d[input_size:input_size + hours_num], p[:hours_num])
  r2s = r2_score(d[input_size:input_size + hours_num], p[:hours_num])
  # print('Max error %.2f' % me)
  # print('Absolute error %.2f' % mae)
  # print('R2 score %.2f' % r2s)
  # print('')

  return me, mae, r2s
 
#run_path = './mlruns/' + str(experiment_id) + '/' + str(run.info.run_id)
#plots_path = run_path + '/plots'
plots_path = './plot'
plot_name = 'a'


mlflow.tensorflow.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")

directory = './data'
os.chdir(directory)
files = os.listdir()
try:
    files.remove('_')
except:
    pass
try:
    files.remove('trained')
except:
    pass
os.chdir('..')
for file in files:
  experiment_id = mlflow.create_experiment(os.path.splitext(file)[0])

  with mlflow.start_run(experiment_id=experiment_id) as run:
    SOURCE_DATA_POWER = f'./data/{file}'
    anomaly_threshold = 0
    anomaly_same_val_len = 3
    input_size = 96
    output_size = 48
    step = 24
    split_size = 0.85
    n_neurons  = 150
    drop = 0.4
    epochs_num = 200

    hours_num = output_size
    step_k = 5

    h = 24 * 7
    sigma = 3
    
    model, data_scaled, total_size, train_size, test_size, scalers, x_train, y_train, x_test, y_test = train_model(SOURCE_DATA_POWER,
                      anomaly_threshold,
                      anomaly_same_val_len,
                      input_size,
                      output_size,
                      step,
                      split_size,
                      n_neurons,
                      drop,
                      epochs_num)
    
    max_k = len(x_test)
    run_path = './mlruns/' + str(experiment_id) + '/' + str(run.info.run_id)
    plots_path = run_path + '/plots'
    os.mkdir(plots_path)
    for k in range(0, max_k, step_k):
      print('k = ' + str(k))
      plot_name = '/plot' + str(k)
      me, mae, r2s = predict_get_metrics(model,
                                         data_scaled,
                                         scalers,
                                         k)

      mlflow.log_artifact(plots_path)
      metrics = {'Max error': me, 'Absolute error': mae,'R2 score': r2s}
      mlflow.log_metrics(metrics)

  mlflow.keras.save_model(model, run_path + '/mlmodel')
