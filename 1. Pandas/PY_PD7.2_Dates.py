# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 18:07:47 2022

@author: rvamsikrishna
"""

#https://www.dataquest.io/blog/datetime-in-pandas/

#%%
#pd.to_datetime accepts a datetime object so you could just do (pandas assumes UTC):
pd.to_datetime(datetime(2020, 5, 11))

#python's datetime is unaware of timezone and will give you a "naive" datetime object 

#You can pass in a tzinfo parameter to the datetime object specifying that the time should be treated as UTC:
from datetime import datetime
import pandas as pd
import pytz

pd.to_datetime(datetime(2020, 5, 11, 0, 0, 0, tzinfo=pytz.UTC).timestamp()*1e9)
 
   
#%%
#1. Exploring Pandas Timestamp and Period Objects

#pandas library provides a DateTime object with nanosecond precision 
#called Timestamp to work with date and time values.

#The Timestamp object derives from the NumPy’s datetime64 data type, 
#making it more accurate and significantly faster than Python’s DateTime object.

import pandas as pd
import numpy as np
from IPython.display import display

print(pd.Timestamp(year=1982, month=9, day=4, hour=1, minute=35, second=10))
print(pd.Timestamp('1982-09-04 1:35.18'))
print(pd.Timestamp('Sep 04, 1982 1:35.18'))

#%%
#If you pass a single integer or float value to the Timestamp constructor,
# it returns a timestamp equivalent to the number of nanoseconds 
#after the Unix epoch (Jan 1, 1970):

print(pd.Timestamp(5000))

#%%
#The Timestamp object inclues many methods and properties that help 
#us access different aspects of a timestamp. Let’s try them:

time_stamp = pd.Timestamp('2022-02-09')
    
print('{}, {} {}, {}'.format(time_stamp.day_name(), \
      time_stamp.month_name(), time_stamp.day, time_stamp.year))

#%%
#While an instance of the Timestamp class represents a single point of time,
# an instance of the Period object represents a period such as a year, 
#a month, etc.

#For example, companies monitor their revenue over a period of a year. 
#Pandas library provides an object called Period to work with periods, 
#as follows:

year = pd.Period('2021')
display(year)
    
#%%
#The Period object provides many useful methods and properties. 
#For example, if you want to return the start and end time of the 
#period, use the following properties:

print('Start Time:', year.start_time)
print('End Time:', year.end_time)

#%%
#To create a monthly period, you can pass a specific month to it, as follows:
month = pd.Period('2022-01')
display(month)
print('Start Time:', month.start_time)
print('End Time:', month.end_time)

#%%
#You also can specify the frequency of the period explicitly with 
#the freq argument.
day = pd.Period('2022-01', freq='D')
display(day)
print('Start Time:', day.start_time)
print('End Time:', day.end_time)

#%%
#We also can perform arithmetic operations on a period object. 
#Let’s create a new period object with hourly frequency and see 
#how we can do the calculations:

hour = pd.Period('2022-02-09 16:00:00', freq='H')
display(hour)
display(hour + 2)
display(hour - 2)

#%%
#We can get the same results using the pandas date offsets:
display(hour + pd.offsets.Hour(+2))
display(hour + pd.offsets.Hour(-2))

#%%
#not working

#To create a sequence of dates, you can use the pandas range_dates() method.
week = pd.date_range('2022-2-7', periods=7)

for day in week:
    print('{}-{}\t{}'.format(day.day_of_week, day.day_name(), day.date()))

#%%

#Creating the Time-Series DataFrame
df = pd.read_csv('https://raw.githubusercontent.com//main/server_util.csv')
display(df.head())
print(df.info())

#%%
df['datetime'] = pd.to_datetime(df['datetime'])
print(df.info())

#%%
df = pd.read_csv('https://raw.githubusercontent.com//main/server_util.csv',\
                 parse_dates=['datetime'])
print(df.head())

#%%
#Convert Column to datetime when Reading a CSV File
df = pd.read_csv('pandas_datetime_example.csv', index_col=0, parse_dates=[3])
#%%
#Convert Column to datetime when Reading an Excel File
df = pd.read_excel('pandas_convert_column_to_datetime.xlsx', 
                 index_col=0, parse_dates=True)

#%%
display(df.datetime.min())
display(df.datetime.max())

#%%
#To select the DataFrame rows between two specific dates, we can create
# a Boolean mask and use the .loc method to filter rows within a 
#certain date range:
    
mask = (df.datetime >= pd.Timestamp('2019-03-06')) \
    & (df.datetime < pd.Timestamp('2019-03-07'))

display(df.loc[mask])

#%%
#Slicing Time Series
#------------------------
#To make Timestamp slicing possible, we need to set the datetime column 
#as the index of the DataFrame. To set a column as an index of a
# DataFrame, use the set_index method:
    
df.set_index('datetime', inplace=True)
print(df)

#%%
#To select all the rows equal to a single index using the .loc method:
print(df.loc['2019-03-07 02:00:00'].head(5))

#%%
#You can select the rows that partially match a specific Timestamp 
#in the index column. Let’s try it:
print(df.loc['2019-03-07'].head(5))

#%%
#The selection string can be any standard date format, let’s look at some examples:
df.loc['Apr 2019']
df.loc['8th April 2019']
df.loc['April 05, 2019 5pm']

#%%
#We can also use the .loc method to slice rows within a date range
display(df.loc['03-04-2019':'04-04-2019'])

#%%
display(df.sort_index().loc['03-04-2019':'04-04-2019'])

#%%
#The DateTimeIndex Methods
#--------------------------
#Some pandas DataFrame methods are only applicable on the DateTimeIndex. 
#We’ll look at some of them in this section, but first,
# let’s make sure our DataFrame has a DateTimeIndex:

print(type(df.index))

#%%
#To return the data collected at a specific time, regardless of the date,
display(df.at_time('09:00'))

#%%
display(df.between_time('00:00','02:00'))

#%%
#data of 1st 5 business days
display(df.sort_index().first('5B'))

#%%
#data of last one week
display(df.sort_index().last('1W'))

#Notice that the DataFrame must be sorted on its index to ensure 
#these methods work. Let’s try both examples:

#%%
df.sort_index().last('2W')

#%%
#Resampling Time Series Data
#------------------------------
#The logic behind the resample() method is similar to the groupby() method. 
#It groups data within any possible period. Although we can use
# the resample() method for both upsampling and downsampling, 

df[df.server_id == 100].resample('D')['cpu_utilization',\
  'free_memory', 'session_count'].mean()

#%%
df.groupby(df.server_id).resample('M')['cpu_utilization', 'free_memory'].max()

#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(24, 8))
df.groupby(df.server_id).resample('M')['cpu_utilization'].mean()\
.plot.bar(color=['green', 'gray'], ax=ax, title='The Average Monthly CPU Utilization Comparison')


#%%
#https://www.programiz.com/python-programming/datetime
    
#https://www.dataquest.io/blog/python-datetime-tutorial/

#https://www.dataquest.io/blog/datetime-in-pandas/

#https://www.analyticsvidhya.com/blog/2020/05/datetime-variables-python-pandas/




































