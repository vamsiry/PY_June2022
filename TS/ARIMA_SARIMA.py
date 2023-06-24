# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:36:07 2022

@author: rvamsikrishna
"""

#ARIMA and Seasonal ARIMA
#Autoregressive Integrated Moving Averages

#The general process for ARIMA models is the following:

# Visualize the Time Series Data
# Make the time series data stationary
# Plot the Correlation and AutoCorrelation Charts
# Construct the ARIMA Model or Seasonal ARIMA based on the data
# Use the model to make predictions

#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

### Testing For Stationarity
from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from pandas.tseries.offsets import DateOffset

from statsmodels.tsa.arima_model import ARIMA

import statsmodels.api as sm
from statsmodels.api.sm.tsa.statespace import SARIMAX


#%%
df=pd.read_csv('C:/Users/rvamsikrishna/Desktop/PY/Python/TS/ARIMA_SARIMA.csv')

df.head()

df.tail()
#%%

#%%
## Cleaning up the data
df.columns=["Month","Sales"]
df.head()
df.tail()

#%%
idx = df[df['Sales'].isnull()].index.tolist()

#np.where(df['Sales'].isnull())[0]

#%%
df.drop(idx,axis=0,inplace=True)

#%%
df[df['Sales'].isnull()].index.tolist()

#%%
type(df['Month'])

#%%
# Convert Month into Datetime
df['Month']=pd.to_datetime(df['Month'])

df['Month'].dt.year
#%%
df['Month'].dt.month
#%%
df['Month'].dt.second

#%%
df.groupby(df['Month'].dt.year).agg({"count","min","max","mean"})

#%%
df['Month-Year'] = df['Month'].dt.month.astype(str) +'/'+ df['Month'].dt.year.astype(str)

#%%
df.head()

#%%
df.drop('Month-Year',axis=1,inplace=True)

#%%
df.set_index('Month',inplace=True)
df.head()

#%%
df.describe()

#%%
#Step 2: Visualize the Data
#-----------------------------
df.plot()

#%%
### Testing For Stationarity
from statsmodels.tsa.stattools import adfuller

#%%
test_result=adfuller(df['Sales'])

#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    
    
#%%
adfuller_test(df['Sales'])

#%%
# Differencing
#-------------
df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)

df['Sales'].shift(1)

#%%
df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)

df.head(14)

#%%
## Again test dickey fuller test
adfuller_test(df['Seasonal First Difference'].dropna())

#%%
df['Seasonal First Difference'].plot()

#%%
#Auto Regressive Model
from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(df['Sales'])
plt.show()

#%%
# Final Thoughts on Autocorrelation and Partial Autocorrelation
# Identification of an AR model is often best done with the PACF.
# For an AR model, the theoretical PACF “shuts off” past the order of the model. The phrase “shuts off” means that in theory the partial autocorrelations are equal to 0 beyond that point. Put another way, the number of non-zero partial autocorrelations gives the order of the AR model. By the “order of the model” we mean the most extreme lag of x that is used as a predictor.
# Identification of an MA model is often best done with the ACF rather than the PACF.
# For an MA model, the theoretical PACF does not shut off, but instead tapers toward 0 in some manner. A clearer pattern for an MA model is in the ACF. The ACF will have non-zero autocorrelations only at lags involved in the model.
# p,d,q p AR model lags d differencing q MA lags
    
#%%
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)

#%%
# For non-seasonal data
#p=1, d=1, q=0 or 1
from statsmodels.tsa.arima_model import ARIMA

model=ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()

model_fit.summary()

#%%
df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))

#%%
import statsmodels.api as sm

model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()

df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))

#%%
from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]

#%%
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

future_datest_df.tail()

#%%
future_df=pd.concat([df,future_datest_df])

#%%
future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
future_df[['Sales', 'forecast']].plot(figsize=(12, 8)) 

#%%











        


     
    