import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Importing dataset
df = pd.read_csv('MLdata.csv')

X = df.drop(columns = ['Population', 'Year', 'Emigration'])
y = df['Population']


tr_errors = []
val_errors = []


for i in range(1, 51):
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.6, random_state=i)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=i)
    
    lin_regr = LinearRegression(fit_intercept=False).fit(X_train,y_train)

    # Predicting the training and validation sets
    y_pred = lin_regr.predict(X_train)
    y_pred_val = lin_regr.predict(X_val)

    # Calculating the Mean Squared Error
    tr_error = mean_squared_error(y_train,y_pred)
    val_error = mean_squared_error(y_val,y_pred_val)

    tr_errors.append(tr_error)
    val_errors.append(val_error)

    
print(tr_errors)
print(val_errors)

print('Mean of linear regression training data:', np.mean(tr_errors))
print('Mean of linear regression validation data:', np.mean(val_errors))


tr_errors_ridge = []
val_errors_ridge = []

for i in range(1, 51):
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.6, random_state=i)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=i)

    ridge_regr = Ridge(alpha=4, fit_intercept=False).fit(X_train, y_train)

    # Predicting the training and validation sets
    y_pred_train = ridge_regr.predict(X_train)
    y_pred_val = ridge_regr.predict(X_val)

    # Calculating the Mean Squared Error
    tr_error_ridge = mean_squared_error(y_train, y_pred_train)
    val_error_ridge = mean_squared_error(y_val, y_pred_val)
    
    tr_errors_ridge.append(tr_error)
    val_errors_ridge.append(val_error)
    
print(tr_errors_ridge)
print(val_errors_ridge)

print('Mean of Ridge training data:', np.mean(tr_errors_ridge))
print('Mean of Ridge validation data:', np.mean(val_errors_ridge))


test_errors = []


for i in range(1, 51):
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.6, random_state=i)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=i)

    lin_regr = LinearRegression(fit_intercept=False).fit(X_test,y_test)

    # Predicting the testing set
    y_pred_test = lin_regr.predict(X_test)

    # Calculating the Mean Squared Error
    test_error = mean_squared_error(y_test,y_pred_test)
    test_errors.append(test_error)

print(test_errors)

print('Mean of linear regression testing data:', np.mean(test_errors))




