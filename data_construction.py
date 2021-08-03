"""Construct the clean data set"""

import pandas as pd
from pathlib import PurePath
import numpy as np
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.interpolate import interp1d
from sklearn.svm import SVR


#========================================================================#
# interpolation functions                                                #
#========================================================================#

def det_interp(x, kind='linear'):
    """ A helper function for deterministic time seres interpolation

    Args
    ----
    x -- a dummy variable for the dataframe's columns

    kwargs
    ------
    kind -- one of scipy.interpolate.inter1d kwargs

    return
    ------
    interpolated values of the whole time series

    """

    index = pd.Series(np.arange(x.shape[0]), index=x.index)

    notnull = pd.notnull(x)

    t = index[notnull]
    
    y = x[notnull]

    f = interp1d(t.values, y.values, kind=kind)

    return pd.Series(f(index), index=x.index, name=x.name)


def ml_interp(x, model, **model_kwargs):
    """ A helper function for ML time seres interpolation

    Args
    ----
    x -- a dummy variable for the dataframe's columns
    model -- a scikit learn model class
    model_kwargs -- tuple of kwargs to pass to the model constructor

    
    return
    ------
    interpolated values of the whole time series

    """

    index = pd.Series(np.arange(x.shape[0]), index=x.index)

    notnull = pd.notnull(x)

    t = index[notnull].values.reshape(-1, 1)
    
    y = x[notnull]
    
    regr = model(**model_kwargs)

    regr.fit(t, y)

    yhat = regr.predict(index.values.reshape(-1, 1))

    return pd.Series(yhat, index=x.index, name=x.name)



def main():

    #=====================================================================#
    # Data import                                                         #
    #=====================================================================#


    root = PurePath() 
    raw_data = root / 'raw_data'
    sentiment = root / 'sentiment_analysis'

    # files
    economics_file = 'economics.csv'
    yields_file = 'FED-SVENY.csv'
    sentiment_file = root / 'daily_sentiment_score.csv'

    # import data
    economics = pd.read_csv(raw_data / economics_file)
    yields = pd.read_csv(raw_data / yields_file)
    sent = pd.read_csv(sentiment / sentiment_file)


    #=====================================================================#
    # clean data                                                          #
    #=====================================================================#

    economics.index = pd.to_datetime(economics['sasdate'], 
            format="%m/%d/%Y")
    economics = economics.iloc[:,1:] # drop date column

    # nan strategy is to drop as of now
    economics = economics[~(economics.apply(np.isnan)).apply(any, axis=1)]
    economics = economics.iloc[:-9,:] # done by inspection


    yields.index = pd.to_datetime(yields['Date'], format="%Y-%m-%d")
    yield_col = ['SVENY10']

    yields = yields[yield_col]

    sent.index = pd.to_datetime(sent['date'], format="%Y-%m-%d")

    sent = sent.drop('date', axis=1)

    #=====================================================================#
    # Join data by date                                                   #
    #=====================================================================#

    # Right now we will move econ dates forward to next trading date
    bday_us = pd.offsets.CustomBusinessDay(
            calendar=USFederalHolidayCalendar())

    economics.index = economics.index +  bday_us

    # 04/01/1999 gets moved when to 04/02/1999 it shouldn't
    as_list = economics.index.tolist()
    idx = as_list.index(dt.datetime(1999, 4, 2))
    as_list[idx] = dt.datetime(1999, 4, 1)
    economics.index = as_list

    full = pd.concat([economics, yields], axis=1, join="outer")

    # Join the tweets
    full = pd.concat([full, sent], axis=1, join="outer")

    #========================================================================#
    # reclean                                                                #
    #========================================================================#

    # drop everything before first appearance of rpi

    inds = full[pd.notnull(full['RPI'])].index.values
    full = full.iloc[full.index.get_loc(inds[0]):,:]

    # drop everything after the last rpi
    full = full.iloc[:full.index.get_loc(inds[-1])+1,:]

    # drop weird generated column 
    full = full.drop('Unnamed: 0', axis=1)

    # reorder columns 
    cols = full.columns.tolist()
    cols = cols[:-2] + cols[-2:][::-1]
    full = full[cols]

    # drop na's added by sentiment score
    full = full[full['SVENY10'].notna()]

    # replace null sentiment score with 0.5
    full['probability'].fillna(0.5, inplace=True)

    # save the file in the data folder
    full.to_csv(raw_data / 'cleaned_full_data.csv')

    return 0


#econ_interp = full.iloc[:,:-1].apply(det_interp, axis=0)
#full = pd.concat([econ_interp, full.iloc[:,-1]], axis=1, join="outer")

#test = ml_interp(full['RPI'], SVR, kernel='rbf', epsilon=0.1)

#econ_ml = full.iloc[:,:-1].apply(ml_interp, axis=0, 
#        args=(SVR,), kernel='rbf', epsilon=0.1)
#full = pd.concat([econ_ml, full.iloc[:,-1]], axis=1, join="outer")

if __name__ == '__main__':
    main()
