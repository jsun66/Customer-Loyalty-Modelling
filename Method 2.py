
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense

# Fix the random number seed to ensure reproducibility
np.random.seed(6)

# Importing the dataset
df = pd.read_excel('Prep_Data.xlsx',sheet_name="Method_2_Train")
df.set_index('OPR_DATE',inplace = True)
X = df.iloc[:,0:12].values
y = df.iloc[:,12].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
y_test = sc_y.transform(y_test.reshape(-1,1))

# Part 2 - FNN Model

# Initialising the FNN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(units = 9, kernel_initializer = 'glorot_uniform', activation = 'relu'))

# Adding the output layer
regressor.add(Dense(units = 1, activation='relu'))

# Optimizing learning rate
#from keras.optimizers import Adam
#optimizer = Adam(lr=0.01)

# Compiling the FNN
regressor.compile(loss='mae', optimizer='adam')

# Fitting the FNN to the Training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)       

# Calculating MAPE of testing set
y_pred = regressor.predict(X_test)
y_predreal=sc_y.inverse_transform(y_pred)
y_testreal=sc_y.inverse_transform(y_test)
np.mean(abs(y_testreal-y_predreal)/y_testreal)

# Calculating MAPE of 2018 July Forecast
forecast = pd.read_excel('Prep_Data.xlsx',sheet_name="Method_2_Forecast")
forecast.set_index('OPR_DATE',inplace = True)
X_forecast = forecast.iloc[:,0:12].values
X_forecast = sc_X.fit_transform(X_forecast)
y_forecast = sc_y.inverse_transform(regressor.predict(X_forecast))
y_real = forecast.iloc[:,12].values.reshape(-1,1)
np.mean(abs(y_real-y_forecast)/y_real)

# Part 3 - XGBoost Model
import xgboost as xgb
regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
xgbModel = regressor.fit(X_train,y_train)

# Calculating MAPE of testing set
y_pred = xgbModel.predict(X_test).reshape(-1,1)
y_predreal=sc_y.inverse_transform(y_pred)
y_testreal=sc_y.inverse_transform(y_test)
np.mean(abs(y_testreal-y_predreal)/y_testreal)

# Calculating MAPE of 2018 July Forecast
forecast = pd.read_excel('Prep_Data.xlsx',sheet_name="Forecast")
forecast.set_index('OPR_DATE',inplace = True)
X_forecast = forecast.iloc[:,0:12].values
X_forecast = sc_X.fit_transform(X_forecast)
y_forecast = sc_y.inverse_transform(xgbModel.predict(X_forecast).reshape(-1,1))
y_real = forecast.iloc[:,12].values.reshape(-1,1)
np.mean(abs(y_real-y_forecast)/y_real)