# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:10:02 2022

@author: rvamsikrishna
"""

#%%
from datetime import datetime
import pandas as pd
import pytz
import calendar

import numpy as np 

#%%
# Create date series using date_range() function
date_series = pd.date_range('08/10/2019', periods = 12, freq ='D')
print(date_series)

#%%
date = pd.to_datetime("8th of sep, 2019")
print(date)
#%%
# Create date series using numpy and to_timedelta() function
date_series = date + pd.to_timedelta(np.arange(12), 'D')
print(date_series)

#%% 
#%%
df = {'Salary':[1000, 2222, 3321, 4414, 5151],
       'Name': ['Pete', 'Steve','Brian','Ryan', 'Jim'],
       'Share':[29.88, 19.05, 8.17,7.3, 6.15],
       'Date':['11/24/2020', '12/21/2019', '10/14/2018', '12/13/2017', '01/08/2017'],
       'Date2': [20120902, 20130413, 20140921, 20140321, 20140321]}

print(df)

df = pd.DataFrame(df)
df.head()

#%%
df.dtypes
#%%
#pd.to_datetime accepts a datetime object so you could just do 
#(pandas assumes UTC):

df['Date'] = pd.to_datetime((df['Date']))
df.dtypes

#pd.to_datetime(datetime(2020, 5, 11, 0, 0, 0, tzinfo=pytz.UTC).timestamp()*1e9)

#%%
df.Date.head()
#%%
df['Date'].dt.year
#%%
df['Date'].dt.month
#%%
df['Date'].dt.day
#%%
df['Date'].dt.hour
#%%
df['Date'].dt.minute
#%%
df['Date'].dt.weekday
#%%
df['Date'].dt.weekday_name
#%%
df['Date'].dt.dayofyear
#%%
pd.Timestamp(pd.datetime(['Date'])).day_name()


#%%
#Convert Date Object into DataFrame Index
#---------------------------------------------
#This can be very helpful for tasks like exploratory data visualization, 
#because matplotlib will recognize that the DataFrame index is a time 
#series and plot the data accordingly.

# Assign date column to dataframe index
df.index = df.date
df.head()


#%%
#%%
#%%
#%%
#convert multiple columns to datetime
df[["col1", "col2", "col3"]] = df[["col1", "col2", "col3"]].apply(pd.to_datetime)

#%%
#Date difference in day when we have dates in 2 columns 
df[['A','B']] = df[['A','B']].apply(pd.to_datetime) #if conversion required
    
df['C'] = (df['B'] - df['A']).dt.days

#%%
#diff b/w dates in mongths
df['nb_months'] = ((df.date2 - df.date1) // np.timedelta64(1, 'M'))

df['nb_months'] = df['nb_months'].astype(int)

#%%
df.assign(
    Months=
    (df.Date2.dt.year - df.Date1.dt.year) * 12 +
    (df.Date2.dt.month - df.Date1.dt.month)
)

#%%
(pd.to_datetime('today').to_period('M') - pd.to_datetime('2022-01-01').to_period('M')).n

#%%



#%%
##Date difference in days by lag of previous payment date 
#using group by and lag of date
import pandas as pd

df_raw_dates = pd.DataFrame({"id": [102, 102, 102, 103, 103, 103, 104],
                             "val": [9,2,4,7,6,3,2],
                             "dates": [pd.Timestamp(2002, 1, 1), pd.Timestamp(2002, 3, 3), pd.Timestamp(2003, 4, 4), pd.Timestamp(2003, 8, 9), pd.Timestamp(2005, 2, 3), pd.Timestamp(2005, 2, 8), pd.Timestamp(2005, 2, 3)]})
print(df_raw_dates)

df_raw_dates.groupby('id').dates.diff().dt.days.fillna(0, downcast='infer')

#%%


