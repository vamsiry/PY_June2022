# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:17:29 2020

@author: rvamsikrishna
"""

#6 Powerful Feature Engineering Techniques For Time Series Data (using Python)

#https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/

#%%
#We don’t have to force-fit traditional time series techniques like 
#ARIMA all the time (I speak from experience!).

# There’ll be projects, such as demand forecasting or click prediction
# when you would need to rely on supervised learning algorithms.
#And there’s where feature engineering for time series comes to the fore.

# =============================================================================
# Sales numbers for the next year 
# Website Traffic
# Competition Position
# Demand of products
# Stock Market Analysis
# Census Analysis
# Budgetary Analysis
# 
# =============================================================================

#%%

#Quick Introduction to Time Series
#------------------------------------
#In a time series, the data is captured at equal intervals and 
#each successive data point in the series depends on its past values.

#We have to forecast the count of people who will take the JetRail 
#on an hourly basis for the next 7 months. 

import pandas as pd

#%%
data = pd.read_csv('Train_SU63ISt.csv')
data.dtypes

data['Datetime'] = pd.to_datetime(data['Datetime'],format='%d-%m-%Y %H:%M')
data.dtypes

#%%
#Feature Engineering for Time Series #1: Date-Related Features

data['year']=data['Datetime'].dt.year 
data['month']=data['Datetime'].dt.month 
data['day']=data['Datetime'].dt.day

data['dayofweek_num']=data['Datetime'].dt.dayofweek  
data['dayofweek_name']=data['Datetime'].dt.weekday_name

data.head()

#%%
#Feature Engineering for Time Series #2: Time-Based Features

data['Hour'] = data['Datetime'].dt.hour 
data['minute'] = data['Datetime'].dt.minute 

data.head()

#%%
#Feature Engineering for Time Series #3: Lag Features
#--------------------------------------------------------

#The lag value we choose will depend on the correlation 
#of individual values with its past values.

#If the series has a weekly trend, which means the value last Monday
# can be used to predict the value for this Monday, you should create
# lag features for seven days.


#We can create multiple lag features as well! Let’s say we want lag 1 to lag 7
#we can let the model decide which is the most valuable one. 
#So, if we train a linear regression model, it will assign appropriate
# weights (or coefficients) to the lag features:


data['lag_1'] = data['Count'].shift(1)
data = data[['Datetime', 'lag_1', 'Count']]
data.head()

data['lag_1'] = data['Count'].shift(1)
data['lag_2'] = data['Count'].shift(2)
data['lag_3'] = data['Count'].shift(3)
data['lag_4'] = data['Count'].shift(4)
data['lag_5'] = data['Count'].shift(5)
data['lag_6'] = data['Count'].shift(6)
data['lag_7'] = data['Count'].shift(7)

data = data[['Datetime', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'Count']]
data.head(10)

#%%
#There is more than one way of determining the lag at which the correlation 
#is significant. For instance, we can use the ACF (Autocorrelation Function) 
#and PACF (Partial Autocorrelation Function) plots.


#ACF: The ACF plot is a measure of the correlation between the time series and the lagged version of itself
#PACF: The PACF plot is a measure of the correlation between the time series with a lagged version of itself but after eliminating the variations already explained by the intervening comparisons

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(data['Count'], lags=10)
plot_pacf(data['Count'], lags=10)


#The partial autocorrelation function shows a high correlation with the
# first lag and lesser correlation with the second and third lag. 
#The autocorrelation function shows a slow decay, which means that 
#the future values have a very high correlation with its past values.


#%%
#Feature Engineering for Time Series #4: Rolling Window Feature
#---------------------------------------------------------------------

#How about calculating some statistical values based on past values? 
#This method is called the rolling window method because
# the window would be different for every data point.

data['rolling_mean'] = data['Count'].rolling(window=7).mean()
data = data[['Datetime', 'rolling_mean', 'Count']]
data.head(10)

#Similarly, you can consider the sum, min, max value, etc.
# (for the selected window) as a feature and try it out on your own machine.

#Recency in an important factor in a time series. Values closer
#to the current date would hold more information.

#Thus, we can use a weighted average, such that higher weights are
# given to the most recent observations. Mathematically, weighted 
#average at time t for the past 7 values would be:
#w_avg = w1*(t-1) + w2*(t-2) + .  .  .  .  + w7*(t-7)


#%%
#Feature Engineering for Time Series #5: Expanding Window Feature
#------------------------------------------------------------------

#This is simply an advanced version of the rolling window technique. 
#In the case of a rolling window, the size of the window is constant 
#while the window slides as we move forward in time. Hence, we consider 
#only the most recent values and ignore the past values.

#The idea behind the expanding window feature is that it takes all 
#the past values into account.

data['expanding_mean'] = data['Count'].expanding(2).mean()
data = data[['Datetime','Count', 'expanding_mean']]
data.head(10)


#%%
#Feature Engineering for Time Series #6: Domain-Specific Features
#-----------------------------------------------------------------

#Having a good understanding of the problem statement, clarity of 
#the end objective and knowledge of the available data is essential to
# engineer domain-specific features for the model.


#Below is the data provided by a retailer for a number of stores and products. 
#Our task is to forecast the future demands for the products. We can come up 
#with various features, like taking a lag or averaging the past values, 
#among other things.

#But hold on. Let me ask you a question – would it be the right way to build
# lag features from lag(1) to lag(7) throughout the data?


#Certainly not! There are different stores and products, and the demand
# for each store and product would be significantly different.

# In this case, we can create lag features considering the 
#store-product combination. 

#Moreover, if we have knowledge about the products and the trends
# in the market, we would be able to generate more accurate (and fewer) features.

#Here’s what I mean – are the sales affected by the weather on the day? 
#Will the sales increase/decrease on a national holiday? 
#If yes, then you can use external datasets and include the list
# of holidays as a feature.


#%%
#Validation Technique for Time Series
#=--------------------------------------

#For the traditional machine learning problems, we randomly select 
#subsets of data for the validation and test sets. But in these cases, 
#each data point is dependent on its past values. If we randomly 
#shuffle the data, we might be training on future data and predicting 
#the past values!

#It is important that we carefully build a validation set when 
#working on a time series problem, without destroying the 
#sequential order within the data.

import pandas as pd
data = pd.read_csv('Train_SU63ISt.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'],format='%d-%m-%Y %H:%M')
data['Datetime'].min(), data['Datetime'].max(), (data['Datetime'].max() -data['Datetime'].min())

data.index = data.Datetime
Train=data.loc['2012-08-25':'2014-06-24'] 
valid=data.loc['2014-06-25':'2014-09-25']


#Great! We have the train and validation sets ready. 
#You can now use these feature engineering techniques and 
#build machine learning models on this data!
















