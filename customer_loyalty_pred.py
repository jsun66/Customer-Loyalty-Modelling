# Part 1 - Data Preprocessing

# Import libraries
import pandas as pd
import numpy as np
from numpy import array
import keras
from keras.models import Sequential
from keras.layers import Dense
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Fix the random number seed to ensure reproducibility
np.random.seed(7)

# Import dataset
df = pd.read_csv('dataset.csv')

# Create dummy variables for categorical features
dummies_Gender = pd.get_dummies(df['Gender'])
dummies_Geo = pd.get_dummies(df['Geography'])
dummies_Exited = pd.get_dummies(df['Exited'])
df=df.join(dummies_Gender)
df=df.join(dummies_Geo)
df=df.join(dummies_Exited)
df=df.drop(['Gender','Geography','F','Quebec','No','Exited'],axis=1)

# Split dataset into training and test set
Y=df['Yes']
X=df.drop(['Yes'],axis=1)
train_x, test_x, train_y, test_y = train_test_split(X,Y, train_size=0.8)

# Feature Scaling
sc_X = MinMaxScaler()
train_x = sc_X.fit_transform(train_x)
test_x = sc_X.fit_transform(test_x)

# Part 2 - Binary Classification Models

# Logistic regression model
def train_logistic_regression(train_x, train_y):
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_x, train_y)
    return logistic_regression_model

trained_logistic_regression_model = train_logistic_regression(train_x, train_y)
pred_y = trained_logistic_regression_model.predict(test_x) 

print(confusion_matrix(test_y, pred_y)) 
print(classification_report(test_y, pred_y))

logit_cv_scores = cross_val_score(trained_logistic_regression_model, test_x, test_y, scoring='f1', cv=5)
logit_cv_mean = np.mean(logit_cv_scores)

# SVM model
from sklearn.svm import SVC  

svclassifier = SVC(kernel='rbf',degree=8)  
svclassifier.fit(train_x, train_y) 
pred_y = svclassifier.predict(test_x)
  
print(confusion_matrix(test_y, pred_y)) 
print(classification_report(test_y, pred_y))

svm_cv_scores = cross_val_score(svclassifier, test_x, test_y, scoring='f1', cv=5)
svm_cv_mean = np.mean(svm_cv_scores)

# FNN model
classifier = Sequential()
classifier.add(Dense(6, activation='relu'))
classifier.add(Dense(6, activation='relu'))
classifier.add(Dense(units = 1, activation = 'relu'))
classifier.compile(loss='binary_crossentropy', optimizer='adam',
                   metrics = ['accuracy'])
history = classifier.fit(train_x, array(train_y), batch_size=25, epochs = 200)

thes=0.50
pred_y = classifier.predict(test_x)  
pred_y[pred_y>=thes]=1
pred_y[pred_y<thes]=0

print(confusion_matrix(test_y, pred_y)) 
print(classification_report(test_y, pred_y))

# XGBoost model
import xgboost as xgb

model1 = xgb.XGBClassifier()
model2 = xgb.XGBClassifier(n_estimators=120, max_depth=5, min_rows=3, learning_rate=0.05, 
                           sample_rate=0.8, col_sample_rate= 0.8, col_sample_rate_per_tree=0.8, 
                           score_tree_interval=5)

train_model1 = model1.fit(train_x, train_y)
train_model2 = model2.fit(train_x, train_y)

pred1 = train_model1.predict(test_x)
pred2 = train_model2.predict(test_x)

print(classification_report(test_y, pred1))
print(classification_report(test_y, pred2))
print(confusion_matrix(test_y, pred1))
print(confusion_matrix(test_y, pred2))

xgb_cv_scores = cross_val_score(model1, test_x, test_y, scoring='f1', cv=5)
xgb_cv_mean = np.mean(xgb_cv_scores)

# Save XGBoost Model1 (best performer)
import pickle
filename = 'finalized_model.sav'
pickle.dump(model1, open(filename, 'wb'))