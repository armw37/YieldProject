import pandas as pd
from pathlib import PurePath
import numpy as np
import data_construction as dc
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

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

# test vs train
tscv = TimeSeriesSplit(n_splits=5, test_size=60)


#========================================================================#
# Lasso                                                                  #
#========================================================================#

X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values

# standardize data
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

# hyperparameter grid
alpha = np.linspace(0.01, 1, 50)

# build models
models = [Lasso(alpha=a, max_iter=5000) for a in alpha]

error = np.ones((alpha.shape[0], 5))

# fit and test the models in a validation loop 
for j, (train_index, test_index) in enumerate(tscv.split(X)):
    # set up data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    for i, m in enumerate(models):
        m.fit(X_train, y_train) # fit
        error[i,j] = mse(y_test, m.predict(X_test)) # predict
        
# mean over validation sets
error = np.mean(error, axis=1)

# get best performing model 
best_index = np.argmin(error)
best_model = models[best_index]
print('best_alpha = ', alpha[best_index])
print('MSE = ', error[best_index])

#========================================================================#
# prinitng results                                                       #
#========================================================================#

# longest column name
max_col_name = max([len(col) for col in df.columns])

# adjust spacing
spacer = max_col_name - 7
if max_col_name - 7 < 0:
    spacer = 0
    
# header 
print('Feature' + ' ' * spacer + '   ' + 'Weight')
print('=======' + ' ' * spacer + '   ' + '======')

# format string
fs = '{0:<' + str(max_col_name) + 's}   {1:<.3f}' 

# sort
weights = np.absolute(best_model.coef_)
index = np.flip(np.argsort(weights))

# print top 10 
for i in index[:10]:
    print(fs.format(df.iloc[:,:-1].columns[i], weights[i]))

print('intercept =', best_model.intercept_)

print('# of non-zero weights =', np.sum(np.where(weights > 0, 1, 0)))

