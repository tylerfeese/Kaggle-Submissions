# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:27:50 2020

@author: Tyler
"""

#importing modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

#assigning datasets
test = pd.read_csv(r'C:\Users\Tyler\Desktop\Projects\Kaggle\Store Item Demand Forecasting\test.csv')
train = pd.read_csv(r'C:\Users\Tyler\Desktop\Projects\Kaggle\Store Item Demand Forecasting\train.csv')

#creating DMatrix on training and test data
dtrain = xgb.DMatrix(data = train[['store','item']],
                     label = train['sales'])
dtest = xgb.DMatrix(data = test[['store','item']])
#defining xgboost parameters
params = {'objective':'reg:linear',
          'max_depth':8,
          'silent':1}

#training model
model = xgb.train(params=params, 
                    dtrain=dtrain)

train_pred = model.predict(dtrain)
test_pred = model.predict(dtest)

#evaluation metrics
mse_train = mean_squared_error(train['sales'],
                               train_pred)

submission = pd.DataFrame()

submission['id'] = test['id']
submission['sales'] = test_pred

submission.to_csv(r'C:\Users\Tyler\Desktop\Projects\Kaggle\Store Item Demand Forecasting\submission.csv')
