import pandas as pd
from pathlib import PurePath
import numpy as np
import data_construction as dc
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

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

# drop gs10
df = df.drop(columns=['GS10',])

# shift time series by 1
df['SVENY10_1'] = df['SVENY10'].shift(-1)
df_1 = df.dropna()

# shift time series by 60
df = df.drop(['SVENY10_1'], axis=1)
df['SVENY10_60'] = df['SVENY10'].shift(-60)
df_60 = df.dropna()

# test vs train
tscv = TimeSeriesSplit(n_splits=5, test_size=60)


#========================================================================#
# Lasso 1                                                                #
#========================================================================#

X_1, y_1 = df_1.iloc[:,:-1].values, df_1.iloc[:,-1].values


# hyperparameter grid
alpha = np.linspace(0.01, 1, 5)

# build models
models_1 = [Lasso(alpha=a, max_iter=5000) for a in alpha]

error = np.ones((alpha.shape[0], 5))

# fit and test the models in a validation loop 
for j, (train_index, test_index) in enumerate(tscv.split(X_1)):
    
    # set up data
    X_train, X_test = X_1[train_index], X_1[test_index]
    y_train, y_test = y_1[train_index], y_1[test_index]

    # standardize data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    for i, m in enumerate(models_1):
        m.fit(X_train, y_train) # fit
        #predict
        error[i,j] = mse(y_test, m.predict(scaler.transform(X_test)))
        
# mean over validation sets
error = np.mean(error, axis=1)

# get best performing model 
best_index = np.argmin(error)
best_model_1 = models_1[best_index]
print('1 Day Lag')
print('best_alpha = ', alpha[best_index])
print('MSE = ', error[best_index])


#========================================================================#
# Lasso 60                                                               #
#========================================================================#

X_60, y_60 = df_60.iloc[:,:-1].values, df_60.iloc[:,-1].values

# standardize data
scaler = StandardScaler().fit(X_60)
X_60 = scaler.transform(X_60)

# hyperparameter grid
alpha = np.linspace(0.01, 1, 5)

# build models
models_60 = [Lasso(alpha=a, max_iter=10000) for a in alpha]

error60 = np.ones((alpha.shape[0], 5))

# fit and test the models in a validation loop 
for j, (train_index, test_index) in enumerate(tscv.split(X_60)):
    # set up data
    X_train, X_test = X_60[train_index], X_60[test_index]
    y_train, y_test = y_60[train_index], y_60[test_index]
    for i, m in enumerate(models_60):
        m.fit(X_train, y_train) # fit
        error60[i,j] = mse(y_test, m.predict(X_test)) # predict
        
# mean over validation sets
error60 = np.mean(error60, axis=1)

# get best performing model 
best_index = np.argmin(error60)
best_model_60 = models_60[best_index]
print('60 Day Lag')
print('best_alpha = ', alpha[best_index])
print('MSE = ', error60[best_index])


#========================================================================#
# saving for prediction plots result                                     #
#========================================================================#


fig, (ax1, ax2) = plt.subplots(2, 1)
# make a little extra space between the subplots
fig.subplots_adjust(hspace=0.5)

# actual data
true = df.iloc[-180:,-2]
t = np.arange(0, true.shape[0], 1)

# 1 period lagged data
scaler_1 = StandardScaler().fit(X_1[:-60])
X_1 = scaler_1.transform(X_1)

# 60 period lagged data
scaler_60 = StandardScaler().fit(X_60[:-60])
X_60 = scaler_60.transform(X_60)

# trained data
yhat_1_train = best_model_1.predict(X_1[-180:-60,:])
yhat_60_train = best_model_60.predict(X_60[-180:-60,:])

# tested data
yhat_1_test = best_model_1.predict(X_1[-60:,:])
yhat_60_test = best_model_60.predict(X_60[-60:,:])


ax1.plot(t[:120], true[:120], label='Actual', color='tab:blue')
ax1.plot(t[120:], true[120:], color='tab:blue')
ax1.plot(t[:120], yhat_1_train, label='Trained', color='tab:orange')
ax1.plot(t[120:], yhat_1_test, label='Predicted', color='tab:green')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('10Y Yield (%)')
ax1.grid(True)
ax1.set_title('1 Day Lag')
ax1.legend()

ax2.plot(t[:120], true[:120], label='Actual', color='tab:blue')
ax2.plot(t[120:], true[120:], color='tab:blue')
ax2.plot(t[:120], yhat_60_train, label='Trained', color='tab:orange')
ax2.plot(t[120:], yhat_60_test, label='Predicted', color='tab:green')
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('10Y Yield (%)')
ax2.grid(True)
ax2.set_title('60 Day Lag')
ax2.legend()

plt.show(block=True)
