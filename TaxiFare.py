# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:59:37 2020

@author: Tyler
"""

#importing modules for project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

#loading train and test data to dataframes
train = pd.read_csv(r'C:\Users\Tyler\Desktop\Projects\Kaggle\New York Taxi Fare Prediction\train.csv')
test = pd.read_csv(r'C:\Users\Tyler\Desktop\Projects\Kaggle\New York Taxi Fare Prediction\test.csv')

#creation of rmse to match with kaggle scoring
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#creation of hour columns from pickup_datetime
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
train['hour'] = train['pickup_datetime'].dt.hour
test['hour'] = test['pickup_datetime'].dt.hour

#function for calculating distance based off of dropoff/pickup lat/longs
def haversine(lat_start, long_start, lat_end, long_end):
    
    #radius of the Earth
    R = 6373.0
    
    #lat-long coordinates
    lat1 = math.radians(lat_start)
    long1 = math.radians(long_start)
    lat2 = math.radians(lat_end)
    long2 = math.radians(long_end)
    
    #change in coordinates
    d_lat = lat2 - lat1
    d_long = long2 - long1
    
    #haversine formula
    a = math.sin(d_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_long / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return(distance)

#adding distance columns to train/test datasets
test['distance'] = test.apply(lambda row:
    haversine(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']),
    axis=1)
train['distance'] = train.apply(lambda row:
    haversine(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']),
    axis=1)

#creating validation set from training data
validation_train, validation_test = train_test_split(train,
                                                     test_size = 0.3,
                                                     random_state = 123)
