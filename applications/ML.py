import pandas as pd 
import os
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from flask import url_for

def model_Air():
  abspth = os.path.abspath('Air_Water_prediction')
  pth=abspth+'/static/city_day.csv'
  df = pd.read_csv(pth)
  df_cleaned = df.dropna(subset=['AQI'])
  df_cleaned=df_cleaned.iloc[:,:16]
  Y = df_cleaned['AQI']
  Y.columns=['AQI']
  df= df_cleaned.iloc[:,2:15]
  df_cleanedX = df_cleaned.iloc[:,2:14]
  df_cleanedX = df_cleanedX.drop(columns=['Benzene'])

  X_train,X_test,Y_train,Y_test = train_test_split(df_cleanedX,Y,test_size=0.2,shuffle=True,random_state=35)

  knn = KNNImputer()
  cols=['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Toluene','Xylene']
  X_train[cols]=pd.DataFrame(knn.fit_transform(X_train),columns=cols,index=X_train.index)
  X_test[cols]=pd.DataFrame(knn.transform(X_test),columns=cols,index=X_test.index)

  ss =StandardScaler()
  cols=['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Toluene','Xylene']
  X_train[cols]=pd.DataFrame(ss.fit_transform(X_train),columns=cols,index=X_train.index)
  X_test[cols]=pd.DataFrame(ss.transform(X_test),columns=cols,index=X_test.index)

  sgd =SGDRegressor(loss='squared_error',penalty='l2', alpha=0.00001, l1_ratio=0.25, fit_intercept=True, max_iter=10000, tol=0.00001, shuffle=True, epsilon=0.1,
                    random_state=42, learning_rate='invscaling', eta0=0.1, power_t=0.25, early_stopping=False, warm_start=False, average=False)
  sgd.fit(X_train,Y_train)
  return sgd,ss

def model_Water():
  abspth = os.path.abspath('Air_Water_prediction')
  pth=abspth+'/static/waterQuality1.csv'
  df = pd.read_csv(pth)
  print(df.shape)
  df = df[df['is_safe'] != '#NUM!']
  print(df.shape)
  Y=df['is_safe']
  X=df.drop(columns=['is_safe'])
  X_train, X_test, y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
  ros = RandomOverSampler(random_state=42)
  X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
  dt = DecisionTreeClassifier(criterion='gini', splitter='best', min_samples_split=4, min_samples_leaf=2, 
                                        max_leaf_nodes=3,min_impurity_decrease=0.01,max_depth=1, class_weight=None,random_state=35)
  ada = AdaBoostClassifier(estimator=dt, n_estimators=50, learning_rate=1.0, algorithm='SAMME', random_state=35)
  ada.fit(X_train_resampled, y_train_resampled)
  return ada

def predict_Air(X,sgd,ss):
  #0-50: Green (Good) 51-100: Yellow (Moderate) 101-150: Orange (Unhealthy for Sensitive Groups) 
  # 151-200: Red (Unhealthy) 201-300: Purple (Very Unhealthy) 301-500: Maroon (Hazardous)
  X=ss.transform(X)
  Y_pred = sgd.predict(X)
  if Y_pred >=0 and Y_pred<=50 :
    return 'Good Air', Y_pred
  elif Y_pred >= 51 and Y_pred <= 100:
    return 'Moderate Air ', Y_pred
  elif Y_pred >= 101 and Y_pred <= 150:
    return 'Unhealthy Air for sensitive people', Y_pred
  elif Y_pred >= 151 and Y_pred <= 200:
    return 'Unhealthy Air', Y_pred
  elif Y_pred >= 201 and Y_pred <= 300:
    return 'Very Unhealthy Air', Y_pred
  elif Y_pred >= 301 and Y_pred <= 500:
    return 'Hazardous Air', Y_pred
  else:
    return 'Severe Air Conditions', Y_pred 

def predict_Water(X,ada):
  Y=ada.predict(X)
  print(Y[0])
  if int(Y[0])==1 :
    return 'Drinkable Water'
  else:
    return 'Unhealthy Water'
  
