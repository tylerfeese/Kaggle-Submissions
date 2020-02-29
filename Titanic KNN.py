#Manipulation Packages
import numpy as np
import pandas as pd

#Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns

#Machine Learning Packages and modules
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#Aliasing provided data
train = pd.read_csv(r'C:\Users\Tyler\Desktop\Personal Files\Kaggle\train.csv')
test =  pd.read_csv(r'C:\Users\Tyler\Desktop\Personal Files\Kaggle\test.csv')

#finding NaNs by column
def find_missing_values(df, columns):
    #returns number of rows in each column with NaNs
    missing_vals = {}
    df_length = len(df)
    for column in columns:
        total_column_values = df[column].value_counts().sum()
        missing_vals[column] = df_length - total_column_values
    return missing_vals

missing_values = find_missing_values(train, columns=train.columns)
missing_values

#Creating new dataframe for fitting cleaning of data and fitting of model
df_train = pd.DataFrame()
df_train['Survived'] = train['Survived']
df_train['Pclass'] = train['Pclass']
df_train['Sex'] = train['Sex']
df_train['Age'] = train['Age']
df_train['Parch'] = train['Parch']
df_train['Embarked'] = train['Embarked']

#Changing Values of Sex from text to numbers (females as 1, males as 0)
df_train.Sex = np.where(df_train['Sex'] == 'female', 1, 0)

#Changing values of Age to average
train['Age'].mean()
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

#Dropping two rows of Embarked without values
df_train = df_train.dropna(subset=['Embarked'])

#Converting Embarked Values to numbers
df_train = df_train.apply(LabelEncoder().fit_transform)

#New dataframe 2 and splitting of data
model_df = df_train
X_train = model_df.drop('Survived',axis=1)
y_train = model_df.Survived

#Algorithm function, runs inputted algorithm using training data and provides accuracy
def fit_model(algo, X_train, y_train, cv):
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    
    train_pred = model_selection.cross_val_predict(algo, X_train, y_train, cv=cv, n_jobs=-1)
    
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    
    return train_pred, acc, acc_cv
#Testing accuracy KNearestNeighbors Classifier
train_pred_knn, acc_knn, acc_cv_knn = fit_model(KNeighborsClassifier(), X_train, y_train, 10)

#Testing Accuracy of different number of neighbors
neighbors =np.arange(1,10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    train_accuracy[i] = knn.score(X_train,y_train)
    
plt.title('k-NN: Varying Number of Neighbors Test of Accuracy')
plt.plot(neighbors, train_accuracy, label='Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#Initiating a KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

#Specifying which columns we want from training data to reformat test data
wanted_test_columns = X_train.columns
predictions = knn.predict(test[wanted_test_columns].apply(LabelEncoder().fit_transform))

submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = predictions

#Exporting to csv
submission.to_csv(r'C:\Users\Tyler\Desktop\Personal Files\Kaggle\knn_submission.csv', index=False)