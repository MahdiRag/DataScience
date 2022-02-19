
import pandas as pd
import numpy as np
import json
import os
import sys
import yfinance
from datetime import date
from datetime import timedelta

today_date = str(pd.to_datetime(date.today()))[0:10]

df = yfinance.download(tickers="BTC-USD", interval='1d', start="2019-01-01", end=today_date).reset_index()
data = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
normalised_data = sc.fit_transform(data.iloc[:,1:])


X_train_60 = []
y_train_60 = []
for i in range(60, len(normalised_data)):
    X_train_60.append(normalised_data[i-60:i, :])
    y_train_60.append(normalised_data[i,0:4])
X_train_60, y_train_60 = np.array(X_train_60), np.array(y_train_60)

# Reshaping
X_train = np.reshape(X_train_60, (X_train_60.shape[0], X_train_60.shape[1], X_train_60.shape[2]))
y_train = y_train_60


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout    

regressor = Sequential()    
    
regressor.add(LSTM(units = 500, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2]))) 
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 500, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 500, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 500))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 4))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = 'accuracy')
regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)


data_test = data.append(data.iloc[-1, :])

sc = MinMaxScaler(feature_range = (0, 1))
normalised_data = sc.fit_transform(data_test.iloc[:,1:])

X_test_60 = []
for i in range(60, len(normalised_data)):
    X_test_60.append(normalised_data[i-60:i, :])
    
X_test_60 = np.array(X_test_60)

# Reshaping
X_test = np.reshape(X_test_60, (X_test_60.shape[0], X_test_60.shape[1], X_test_60.shape[2]))

predictions = pd.DataFrame(sc.inverse_transform(regressor.predict(X_test[-1:])))
predictions.columns = ['Open', 'High', 'Low', 'Close']
tomorrow = pd.to_datetime(date.today() + timedelta(days=1))
predictions.insert(0, 'Date', tomorrow)
print(predictions)


















