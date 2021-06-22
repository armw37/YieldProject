"""Construct the clean data set"""

import pandas as pd
from pathlib import PurePath
import numpy as np
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar


#========================================================================#
# Data import                                                            #
#========================================================================#


root = PurePath() 
raw_data = root / 'raw_data'

# files
economics_file = 'economics.csv'
yields_file = 'FED-SVENY.csv'


# import data
economics = pd.read_csv(raw_data / economics_file)
yields = pd.read_csv(raw_data / yields_file)


#========================================================================#
# clean data                                                             #
#========================================================================#

economics.index = pd.to_datetime(economics['sasdate'], format="%m/%d/%Y")
economics = economics.iloc[:,1:] # drop date column

# nan strategy is to drop as of now
economics = economics[~(economics.apply(np.isnan)).apply(any, axis=1)]
economics = economics.iloc[:-9,:] # done by inspection


yields.index = pd.to_datetime(yields['Date'], format="%Y-%m-%d")
yield_col = ['SVENY01',	'SVENY02', 'SVENY03', 'SVENY05', 'SVENY07', 
        'SVENY10', 'SVENY20', 'SVENY30']
yields = yields[yield_col]


#========================================================================#
# Join data by date                                                      #
#========================================================================#

# Right now we will move econ dates forward to next trading date

bday_us = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())

economics.index = economics.index +  bday_us

# 04/01/1999 gets moved when to 04/02/1999 it shouldn't
as_list = economics.index.tolist()
idx = as_list.index(dt.datetime(1999, 4, 2))
as_list[idx] = dt.datetime(1999, 4, 1)
economics.index = as_list

full = pd.concat([economics, yields], axis=1, join="inner")




