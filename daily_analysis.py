"""Analysis for the daily time series"""

import pandas as pd
from pathlib import PurePath
import numpy as np
import data_construction as dc
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error as mse

#========================================================================#
# Plan                                                                   #
#========================================================================#
# 
# import clean data and linear interp for now
#
# fit lasso regression on full data
#
# fit arima on 10Y time series
#
# fit lasso regression on full data 1 period lag
#
# fit lasso regression on full data 60 period lag
#
# PCA feature space
#
# Fit NN on PCA 60 period
#
# Fit NN on PCA 1 period lag
#
# Fit DCNN on PCA 60 period
#
# Fit LSTM on PCA 1 period lag
#
# forecast 60 period and use RMSE as the performance metric
#
#========================================================================#
# data import                                                            #
#========================================================================#

root = PurePath() 
raw_data = root / 'raw_data'

# file
clean_data = 'cleaned_full_data.csv'

# import data
df = pd.read_csv(raw_data / clean_data, index_col=0)

# interpolate and join
test = dc.det_interp(df.iloc[:,1])
econ_interp = df.iloc[:,:-1].apply(dc.det_interp, axis=0)
df = pd.concat([econ_interp, df.iloc[:,-1]], axis=1, join="outer")

# perform time lags
df['SVENY10_1'] = df['SVENY10'].shift(1)
df['SVENY10_60'] = df['SVENY10'].shift(60)
df_lag_1 = df.iloc[:,:-1].dropna().drop(['SVENY10'], axis=1)
df_lag_60 = df.drop(['SVENY10_1', 
    'SVENY10'], axis=1).dropna()
df = df.iloc[:,:-2]

# test vs train
tscv = TimeSeriesSplit(n_splits=5, test_size=60)


#========================================================================#
# Lasso normal                                                           #
#========================================================================#

# this is to understand the important features
feature_model = linear_model.Lasso(alpha=0.1, normalize=True)
feature_model.fit(df.iloc[:,:-1], df.iloc[:,-1])

# print the important features
max_col_name = max([len(col) for col in df.columns])

spacer = max_col_name - 7
if max_col_name - 7 < 0:
    spacer = 0
    
# header 
print('Feature' + spacer + ' : ' + 'Weight')
print('=======' + spacer + '   ' + '======')

# format string
fs = '{0:<' + max_col_name + 's} : {1:<.3f}' 

for i, col in enumerate(df.columns):
    print(fs.format(col, clf.coef_[i]))

print('intercept = ' + clf.intercept_)


#========================================================================#
# ARIMA                                                                  #
#========================================================================#

# ARIMA uses just the time series of interest

# rolling for 1 period ahead forecast 

# arima 60 day shot


#========================================================================#
# Lasso 1 lag                                                           #
#========================================================================#

lasso_1_lag_models = [linear_model.Lasso(alpha=0.01, normalize=True),
    linear_model.Lasso(alpha=0.1, normalize=True), 
    linear_model.Lasso(alpha=0.25, normalize=True),
    linear_model.Lasso(alpha=0.5, normalize=True),
    linear_model.Lasso(alpha=1, normalize=True)]

lasso_1_lag_mse = []

for i, (train_index, test_index) in enumerate(tscv.split(df_lag_1)):
    
    # set up data
    X_train, X_test = df_lag_1.iloc[train_index,:-1], \
            df_lag_1.iloc[test_index,:-1]
    y_train, y_test = df_lag_1.iloc[train_index,-1], \
            df_lag_1.iloc[test_index,-1]

    # train
    lasso_1_lag_models[i].fit(X_train, y_train)

    # test 
    yhat = lasso_1_lag_models[i].predict(X_test)
    lasso_1_lag_mse.append(mse(y_test, yhat))

# choose the best model
best_lasso_1 = np.argmin(lasso_1_lag_mse)
print('Best 1 day lag Lasso model is:', best_lasso_1)

# do some more analysis if we have time (plot prediction vs actual)


#========================================================================#
# Lasso 60 lag                                                           #
#========================================================================#

lasso_60_lag_models = [linear_model.Lasso(alpha=0.01, normalize=True),
    linear_model.Lasso(alpha=0.1, normalize=True), 
    linear_model.Lasso(alpha=0.25, normalize=True),
    linear_model.Lasso(alpha=0.5, normalize=True),
    linear_model.Lasso(alpha=1, normalize=True)]

lasso_60_lag_mse = []

for i, (train_index, test_index) in enumerate(tscv.split(df_lag_60)):
    
    # set up data
    X_train, X_test = df_lag_60.iloc[train_index,:-1], \
            df_lag_60.iloc[test_index,:-1]
    y_train, y_test = df_lag_60.iloc[train_index,-1], \
            df_lag_60.iloc[test_index,-1]

    # train
    lasso_60_lag_models[i].fit(X_train, y_train)

    # test 
    yhat = lasso_60_lag_models[i].predict(X_test)
    lasso_60_lag_mse.append(mse(y_test, yhat))

# choose the best model
best_lasso_60 = np.argmin(lasso_60_lag_mse)
print('Best 60 day lag Lasso model is:', best_lasso_60)

# do some more analysis if we have time (plot prediction vs actual)


#========================================================================#
# PCA feaure space is easy                                               #
#========================================================================#

# sklearn PCA


#========================================================================#
# CNN as supervised 1 day lag                                            #
#========================================================================#

# basically same as lasso

# use the PCA features of the next pushed back periods to predict


#========================================================================#
# CNN 60 as supervised period lag                                        #
#========================================================================#

# basically same as lasso

# use the PCA features of the next pushed back periods to predict

#========================================================================#
# CNN as single shot 60 day                                              #
#========================================================================#

# how far back to go? 60 days maybe 

# use the PCA features of the next pushed back periods to predict


#========================================================================#
# LSTM as single 60 day                                                  #
#========================================================================#

# how far back to go? 60 days maybe

# use the PCA features of the next pushed back periods to predict

