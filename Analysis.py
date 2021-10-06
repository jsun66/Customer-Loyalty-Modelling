# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 23:12:04 2018

@author: MING
"""

# Importing the libraries
import numpy as np
from numpy import array
from numpy import reshape
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#from keras.layers import Dropout

data=pd.read_excel('Loading Summary.xlsx',sheetname="Summer",parse_cols = "B:P")
ResDat2016=pd.read_excel('2009-2016 Site Count.xlsx',sheetname="2016",parse_cols = "B,I")
ResDat2011=pd.read_excel('2009-2016 Site Count.xlsx',sheetname="2011",parse_cols = "B,I")
ResList2016 = ResDat2016['Rate'].tolist()
ResList2011 = ResDat2011['Rate'].tolist()

X=pd.DataFrame(columns=['Load','GDP','Temp','ResRatio'])
Y=pd.DataFrame(columns=['NLoad'])

for i in range (0,227):
    print (i)
    feedername=data.Feeder[i]
    if feedername[0]=="8": 
        if len(feedername)==6:
            feedername="008-0"+feedername[2:6]
        else:
            feedername="00"+feedername
    if feedername[0]=="2": 
        if len(feedername)==8:
            feedername="025-0"+feedername[3:8]
        else:
            feedername="0"+feedername
    ResRatio2016=0.7 #default value        
    ResRatio2011=0.7 #default value 
    if feedername in ResList2016:
        ResRatio2016=ResDat2016[ResDat2016.Rate==feedername].iloc[0,1] #Use "==" and iloc to obtain the residential ratio 
    if feedername in ResList2011:
        ResRatio2011=ResDat2011[ResDat2011.Rate==feedername].iloc[0,1] #Use "==" and iloc to obtain the residential ratio        
    for k in range(0,9):
        row4=array(data.iloc[i,3+k:3+k+4])
        if k<=4: 
            ResRatio=ResRatio2011
        if k>4: 
            ResRatio=ResRatio2016
        if k==0:
            GDP=[2.1,1.3,-5.3]
            MaxT=[32.3,34,33.2]
            MinT=[-26.1,-33.3,-32.4]
        if k==1:
            GDP=[1.3,-5.3,5.1]
            MaxT=[34,33.2,31.4]
            MinT=[-33.3,-32.4,-31.2]
        if k==2:
            GDP=[-5.3,5.1,6.7]
            MaxT=[33.2,31.4,30.4]
            MinT=[-32.4,-31.2,-29.9]
        if k==3:
            GDP=[5.1,6.7,4]
            MaxT=[31.4,30.4,30.5]
            MinT=[-31.2,-29.9,-32.4]
        if k==4:
            GDP=[6.7,4,5.8]
            MaxT=[30.4,30.5,32.8]
            MinT=[-29.9,-32.4,-30.1]
        if k==5:
            GDP=[4,5.8,4.9]
            MaxT=[30.5,32.8,32.2]
            MinT=[-32.4,-30.1,-30.4]
        if k==6:
            GDP=[5.8,4.9,-3.7]
            MaxT=[32.8,32.2,33.6]
            MinT=[-30.1,-30.4,-25.3]
        if k==7:
            GDP=[4.9,-3.7,-3.8]
            MaxT=[32.2,33.6,30.9]
            MinT=[-30.4,-25.3,-26.7]
        if k==8:
            GDP=[-3.7,-3.8,4.9]
            MaxT=[33.6,30.9,33]
            MinT=[-25.3,-26.7,-31.3]
                
        if row4[3]>0 and row4[2]>0 and row4[1]>0 and row4[0]>0:            
            #if (abs(row4[3]-row4[2])/row4[2]<0.2 and abs(row4[2]-row4[1])/row4[1]<0.2 and abs(row4[1]-row4[0])/row4[0]<0.2) or ResRatio<0.7:
            if  ResRatio>0.9:
                X=X.append({'Load':row4[0],'ResRatio':ResRatio,'GDP':GDP[0],'Temp':MaxT[0]},ignore_index=True)
                Y=Y.append({'NLoad':row4[1]},ignore_index=True)
                X=X.append({'Load':row4[1],'ResRatio':ResRatio,'GDP':GDP[1],'Temp':MaxT[1]},ignore_index=True)
                Y=Y.append({'NLoad':row4[2]},ignore_index=True)
                X=X.append({'Load':row4[2],'ResRatio':ResRatio,'GDP':GDP[2],'Temp':MaxT[2]},ignore_index=True)
                Y=Y.append({'NLoad':row4[3]},ignore_index=True)
                
                
Z1=pd.concat([X,Y], axis=1) # directly combine X and Y horizontally 
Z=Z1.drop_duplicates (subset="Load") # drop duplicates by column load
X=Z[['Load','ResRatio','GDP','Temp']]
y=Z[['NLoad']]
                
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)


# Part 2 - Making the ANN

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(units = 4, kernel_initializer = 'he_uniform', activation = 'tanh'))
#regressor.add(Dropout(0.1))

## Adding the second hidden layer
#regressor.add(Dense(units = 58, kernel_initializer = 'he_uniform', activation = 'selu'))
##regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1, kernel_initializer = 'he_uniform', activation = 'selu'))

# Compiling the ANN
#from keras.optimizers import Adam
#optimizer = Adam(lr=0.3)
regressor.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mean_squared_error'])

# Fitting the ANN to the Training set
history = regressor.fit(X_train, y_train, batch_size =12, epochs = 200)
# Plot Metrics
from matplotlib import pyplot
pyplot.plot(history.history['mean_squared_error'])
#pyplot.plot(history.history['Loss'])
pyplot.show()                

y_pred = regressor.predict(X_test)
#y_pred = (y_pred > 0.5)
y_predreal=sc_y.inverse_transform(y_pred)
y_testreal=sc_y.inverse_transform(y_test)
np.mean(abs(y_testreal-y_predreal)/y_testreal)

# Test feature significance
M_test=M_test=array([X_test[5]])
y_pred = regressor.predict(M_test)
print (sc_X.inverse_transform(M_test))
print (sc_y.inverse_transform(y_pred))

 #Part 3 - Making the LSTM - Many to Many

# Use the orignal Z (three rows for every timestep). There are repeated ones but doesn't matter for training.
X=Z1[['Load','ResRatio','GDP','Temp']]
y=Z1[['NLoad']]

XL=array(X).reshape(int(len(X)/3),12)
YL=array(Y).reshape(int(len(X)/3),3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(XL, YL, test_size = 0.2)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

X_train3D=X_train.reshape(X_train.shape[0],3,2)
X_test3D=X_test.reshape(X_test.shape[0],3,2)

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(LSTM(18, input_shape=(X_train3D.shape[1], X_train3D.shape[2])))

# Adding the output layer
#regressor.add(Dense(units = 1, kernel_initializer = 'he_uniform', activation = 'selu'))
regressor.add(Dense(3))

# Compiling the ANN
#from keras.optimizers import Adam
#optimizer = Adam(lr=0.3)
regressor.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mean_squared_error'])

# Fitting the ANN to the Training set
history = regressor.fit(X_train3D, y_train, batch_size=18, epochs = 200)
# Plot Metrics
from matplotlib import pyplot
pyplot.plot(history.history['mean_squared_error'])
#pyplot.plot(history.history['Loss'])
pyplot.show()                

y_pred = regressor.predict(X_test3D)
#y_pred = (y_pred > 0.5)
y_predreal=sc_y.inverse_transform(y_pred)
y_testreal=sc_y.inverse_transform(y_test)
np.mean(abs(y_testreal[:,2]-y_predreal[:,2])/y_testreal[:,2])

#Part 3 - Making the LSTM - Many to One

# Use the orignal Z (three rows for every timestep). There are repeated ones but doesn't matter for training.
X=Z1[['Load','ResRatio','GDP','Temp']]
y=Z1[['NLoad']]

XL=array(X).reshape(int(len(X)/3),12)
YL=array(Y).reshape(int(len(X)/3),3)
YLOne=YL[:,2] #Obtain only the last column as the output

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(XL, YLOne, test_size = 0.2)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
y_test = sc_y.transform(y_test.reshape(-1,1))

X_train3D=X_train.reshape(X_train.shape[0],3,4)
X_test3D=X_test.reshape(X_test.shape[0],3,4)

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the first hidden layer
#regressor.add(LSTM(18, input_shape=(X_train3D.shape[1], X_train3D.shape[2])))
regressor.add(LSTM(units=9, input_shape=(X_train3D.shape[1], X_train3D.shape[2]),kernel_initializer = 'he_uniform', activation = 'tanh'))

# Adding the output layer
#regressor.add(Dense(units = 1, kernel_initializer = 'he_uniform', activation = 'selu'))
regressor.add(Dense(units = 1, kernel_initializer = 'he_uniform', activation = 'selu'))

# Compiling the ANN
#from keras.optimizers import Adam
#optimizer = Adam(lr=0.3)
regressor.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mean_squared_error'])

# Fitting the ANN to the Training set
history = regressor.fit(X_train3D, y_train, batch_size=12, epochs = 800)
# Plot Metrics
from matplotlib import pyplot
pyplot.plot(history.history['mean_squared_error'])
#pyplot.plot(history.history['Loss'])
pyplot.show()                

y_pred = regressor.predict(X_test3D)
#y_pred = (y_pred > 0.5)
y_predreal=sc_y.inverse_transform(y_pred)
y_testreal=sc_y.inverse_transform(y_test)
print (np.mean(abs(y_testreal-y_predreal)/y_testreal))
print (np.mean((y_testreal-y_predreal)/y_testreal))

# Save Trained Model and Weights
regressor.save_weights('model_weights.h5')
with open('model_architecture.json', 'w') as f:
    f.write(regressor.to_json())
    
#Read Trained Model and Weights   
#from keras.models import model_from_json
#      
#with open('model_architecture.json', 'r') as f:
#     regressor = model_from_json(f.read())
#regressor.load_weights('model_weights.h5')
    
