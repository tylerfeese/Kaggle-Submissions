# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:31:10 2019
@author: Tyler
"""
#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor, plot_importance

#labeling train and test data
train = pd.read_csv(r'C:\Users\Tyler Feese\Desktop\Kaggle\Housing Regression\train.csv')
test = pd.read_csv(r'C:\Users\Tyler Feese\Desktop\Kaggle\Housing Regression\test.csv')

#secondary dataframes
traindf = pd.DataFrame(train)
testdf = pd.DataFrame(test)

#Replacing missing data in several columns with means/modes
traindf['LotFrontage'] = traindf['LotFrontage'].fillna(traindf['LotFrontage'].mean())
testdf['LotFrontage'] = testdf['LotFrontage'].fillna(testdf['LotFrontage'].mean())

traindf['Exterior2nd'] = traindf['Exterior2nd'].fillna(traindf['Exterior2nd'].mode()[0])
testdf['Exterior2nd'] = testdf['Exterior2nd'].fillna(testdf['Exterior2nd'].mode()[0])

traindf['BsmtCond'] = traindf['BsmtCond'].fillna(traindf['BsmtCond'].mode()[0])
testdf['BsmtCond'] = testdf['BsmtCond'].fillna(testdf['BsmtCond'].mode()[0])

traindf['BsmtQual'] = traindf['BsmtQual'].fillna(traindf['BsmtQual'].mode()[0])
testdf['BsmtQual'] = testdf['BsmtQual'].fillna(testdf['BsmtQual'].mode()[0])

traindf['FireplaceQu'] = traindf['FireplaceQu'].fillna(traindf['FireplaceQu'].mode()[0])
testdf['FireplaceQu'] = testdf['FireplaceQu'].fillna(testdf['FireplaceQu'].mode()[0])

traindf['GarageType'] = traindf['GarageType'].fillna(traindf['GarageType'].mode()[0])
testdf['GarageType'] = testdf['GarageType'].fillna(testdf['GarageType'].mode()[0])

traindf['GarageFinish'] = traindf['GarageFinish'].fillna(traindf['GarageFinish'].mode()[0])
testdf['GarageFinish'] = testdf['GarageFinish'].fillna(testdf['GarageFinish'].mode()[0])

traindf['GarageQual'] = traindf['GarageQual'].fillna(traindf['GarageQual'].mode()[0])
testdf['GarageQual'] = testdf['GarageQual'].fillna(testdf['GarageQual'].mode()[0])

traindf['GarageCond'] = traindf['GarageCond'].fillna(traindf['GarageCond'].mode()[0])
testdf['GarageCond'] = testdf['GarageCond'].fillna(testdf['GarageCond'].mode()[0])

traindf['MasVnrType'] = traindf['MasVnrType'].fillna(traindf['MasVnrType'].mode()[0])
testdf['MasVnrType'] = testdf['MasVnrType'].fillna(testdf['MasVnrType'].mode()[0])

traindf['MasVnrArea'] = traindf['MasVnrArea'].fillna(traindf['MasVnrArea'].mode()[0])
testdf['MasVnrArea'] = testdf['MasVnrArea'].fillna(testdf['MasVnrArea'].mode()[0])

traindf['BsmtExposure'] = traindf['BsmtExposure'].fillna(traindf['BsmtExposure'].mode()[0])
testdf['BsmtExposure'] = testdf['BsmtExposure'].fillna(testdf['BsmtExposure'].mode()[0])

traindf['BsmtFinType2'] = traindf['BsmtFinType2'].fillna(traindf['BsmtFinType2'].mode()[0])
testdf['BsmtFinType2'] = testdf['BsmtFinType2'].fillna(testdf['BsmtFinType2'].mode()[0])

traindf['BsmtFinType1'] = traindf['BsmtFinType1'].fillna(traindf['BsmtFinType1'].mode()[0])
testdf['BsmtFinType1'] = testdf['BsmtFinType1'].fillna(testdf['BsmtFinType1'].mode()[0])

testdf['BsmtFullBath'] = testdf['BsmtFullBath'].fillna(testdf['BsmtFullBath'].mode()[0])
testdf['BsmtHalfBath'] = testdf['BsmtHalfBath'].fillna(testdf['BsmtHalfBath'].mode()[0])

testdf['MSZoning'] = testdf['MSZoning'].fillna(testdf['MSZoning'].mode()[0])
testdf['SaleType'] = testdf['SaleType'].fillna(testdf['SaleType'].mode()[0])

#Dropping columns with too many nulls
traindf.drop(['Alley','PoolQC','Fence','MiscFeature','GarageYrBlt'],axis=1,inplace=True)
testdf.drop(['Alley','PoolQC','Fence','MiscFeature','GarageYrBlt'],axis=1,inplace=True)

#Handling Categorical columns and getting dummy variables
categories_test = ['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition']
categories_train = ['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition']

testdf1= pd.DataFrame(pd.get_dummies(testdf[categories_test],drop_first=True))
traindf1= pd.DataFrame(pd.get_dummies(traindf[categories_train],drop_first=True))

#Dropping Categorical columns from original dataframe and merging for fitting of models
testdf.drop(['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'],axis=1,inplace=True)
traindf.drop(['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'],axis=1,inplace=True)

jointest = pd.concat([testdf,testdf1],axis=1,join='outer')
jointrain = pd.concat([traindf,traindf1],axis=1,join='outer')

jointest.to_csv(r'C:\Users\Tyler Feese\Desktop\Kaggle\Housing Regression\cleanedtest.csv',index=False)
jointrain.to_csv(r'C:\Users\Tyler Feese\Desktop\Kaggle\Housing Regression\cleanedtrain.csv',index=False)

test_df = pd.read_csv(r'C:\Users\Tyler Feese\Desktop\Kaggle\Housing Regression\cleanedtest.csv',header=0)
train_df = pd.read_csv(r'C:\Users\Tyler Feese\Desktop\Kaggle\Housing Regression\cleanedtrain.csv',header=0)

#Defining of variables for models
y_train = train_df['SalePrice']
X_train = train_df.drop(['SalePrice','Id'],axis=1)
X_train = X_train.drop(['Utilities_NoSeWa',
                   'Condition2_RRAe',
                  'Condition2_RRAn',
                   'Condition2_RRNn',
                   'HouseStyle_2.5Fin',
                   'RoofMatl_CompShg',
                   'RoofMatl_Membran',
                   'RoofMatl_Metal',
                   'RoofMatl_Roll',
                   'Exterior1st_ImStucc',
                   'Exterior1st_Stone',
                   'Exterior2nd_Other',
                   'Heating_GasA',
                   'Heating_OthW',
                   'Electrical_Mix',
                   'GarageQual_Fa'
                   ]
     ,axis=1)

#Intializing of models and splitting of training data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size =.3,random_state=123)
reg = LinearRegression()
dt = DecisionTreeRegressor(max_depth=6,min_samples_leaf=0.05,random_state=123)
rf = RandomForestRegressor(n_estimators=50, random_state=123)
rf.fit(X_train,y_train)
reg.fit(X_train,y_train)
dt.fit(X_train,y_train)
ydt_pred=dt.predict(X_test)
yreg_pred=reg.predict(X_test)
yrf_pred = (rf.predict(X_test))

#Calculation of R Square values of models and RMSE for evaluation
print("R^2 for Regression Tree: {}".format(dt.score(X_test, y_test)))
print("R^2 for Linear Regression: {}".format(reg.score(X_test, y_test)))
print("R^2 for Random Forest: {}".format(rf.score(X_test, y_test)))

ydtrmse = np.sqrt(mean_squared_error(y_test, ydt_pred))
yregrmse = np.sqrt(mean_squared_error(y_test, yreg_pred))
yrfrmse = np.sqrt(mean_squared_error(y_test, yrf_pred))

print("Root Mean Squared Error Linear Regression: {}".format(yregrmse))
print("Root Mean Squared Error Regression Tree: {}".format(ydtrmse))
print("Root Mean Squared Error Random Forest: {}".format(yrfrmse))

#RMSLE function to match with Kaggle scoring
def rmsle(y,y0):
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))

dtrmsle = rmsle(y_test, ydt_pred)
regrmsle = rmsle(y_test, yreg_pred)
rfrmsle = rmsle(y_test, yrf_pred)

print("Root Mean Squared Logged Error Linear Regression: {}".format(regrmsle))
print("Root Mean Squared Logged Error Regression Tree: {}".format(dtrmsle))
print("Root Mean Squared Logged Error Random Forest: {}".format(rfrmsle))
#Scoring of Training Data Accuracy
dtscore = dt.score(X_train,y_train)
regscore = reg.score(X_train,y_train)
rfscore = rf.score(X_train, y_train)
print('Linear Regression Model Score: ', regscore)
print('Regression Tree Model Score: ', dtscore)
print('Random Forest Model Score: ', rfscore)

#Redefining of Test Data and making of predictions on true test data
X_test = test_df.drop(['Id'],axis=1)
#Initializing of missing columns DataFrame to insert to Test DataFrame
#X_test = X_test.join(pd.DataFrame(
 #           {
  #                 'MSZoning_FV':0,
   #                'LotConfig_FR2':0,
    #               'BldgType_Twnhs':0,
     #              'BldgType_TwnhsE':0,
      #             'HouseStyle_1.5Unf':0,
       #            'RoofStyle_Gambrel':0,
        #           'Exterior1st_CBlock':0,
         #          'Exterior1st_CemntBd':0,
          #         'Exterior1st_HdBoard':0,
           #        'Exterior1st_MetalSd':0,
            #       'Exterior2nd_CmentBd':0,
             #      'Exterior2nd_MetalSd':0,
              #     'ExterCond_Fa':0,
               #    'KitchenQual_TA':0,
                #   'Functional_Min2':0,
                 #  'GarageType_Basment':0,
                  # 'SaleType_Wd':0
                   # },index=X_test.index))

predictions = dt.predict(X_test.apply(LabelEncoder().fit_transform))

submission = pd.DataFrame()
submission['Id'] = test_df['Id']
submission['SalePrice'] = predictions


#Exporting to csv
submission.to_csv(r'C:\Users\Tyler Feese\Desktop\Kaggle\Housing Regression\housingprices_submission.csv', index=False)
