# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:52:03 2022

@author: rvamsikrishna
"""

#https://www.analyticsvidhya.com/blog/2021/05/feature-scaling-techniques-in-python-a-complete-guide/

#Why Feature Scaling?

#Feature Scaling is done to normalize the features in the dataset into a finite range.

#for example let’s consider the house price prediction dataset. It will have 
#many features like no. of. bedrooms, square feet area of the house, etc.

#As you can guess, the no. of bedrooms will vary between 1 and 5, but the 
#square feet area will range from 500-2000. This is a huge difference in
# the range of both features.

#Many machine learning algorithms that are using Euclidean distance as a metric
# to calculate the similarities will fail to give a reasonable recognition 
#to the smaller feature, in this case, the number of bedrooms, which in the
# real case can turn out to be an actually important metric.
 
#%%
#%%
#Absolute Maximum Scaling(-1 to +1)
#******************************
#Find the absolute maximum value of the feature in the dataset

#Divide all the values in the column by that maximum value

#If we do this for all the numerical columns, then all their values will 
#lie between -1 and 1

#The main disadvantage is that the technique is sensitive to outliers. 
#Like consider the feature *square feet*, if 99% of the houses have square
# feet area of less than 1000, and even if just 1 house has a square feet 
#area of 20,000, then all those other house values will be scaled down
# to less than 0.05.
    
#I will be working with the sine and cosine functions throughout the article
# and show you how the scaling techniques affect their magnitude. 
#sin() will be ranging between -1 and +1, and 50*cos() will be ranging 
#between -50 and +50.

import numpy as np
x = np.arange(0,20,0.4)
y1 = np.sin(x)
y2 = np.cos(x)*50

import matplotlib.pyplot as plt
plt.plot(x,y1,'red')
plt.plot(x,y2,'blue')

#%%
y1_new = y1/max(y1)
y2_new = y2/max(y2)

plt.plot(x,y1_new,'red')
plt.plot(x,y2_new,'blue')

#See from the graph that now both the datasets are ranging from -1 to +1
# after the scaling.

#%%
#%%
#Min Max Scaling (0-1)--Also called as Normalization
#***************************************************

#In min-max you will subtract the minimum value in the dataset with all 
#the values and then divide this by the range of the dataset(maximum-minimum).
# In this case, your dataset will lie between 0 and 1 in all cases 
#whereas in the previous case, it was between -1 and +1. 
#Again, this technique is also prone to outliers.

y1_new = (y1-min(y1))/(max(y1)-min(y1))
y2_new = (y2-min(y2))/(max(y2)-min(y2))

plt.plot(x,y1_new,'red')
plt.plot(x,y2_new,'blue')

#%%
#%%
#Normalization also called as Min Max Scaling (0-1)
#****************************************************
#Instead of using the min() value in the previous case, in this case,
# we will be using the average() value.

#In scaling, you are changing the range of your data while in normalization 
#you arere changing the shape of the distribution of your data.

y1_new = (y1-np.mean(y1))/(max(y1)-min(y1))
y2_new = (y2-np.mean(y2))/(max(y2)-min(y2))

plt.plot(x,y1_new,'red')
plt.plot(x,y2_new,'blue')


#%%
#%%
#Robust Scaling --- Almost similar to Normalization more robust for outliers
#****************************************************************************

#In this method, you need to subtract all the data points with the 
#median value and then divide it by the Inter Quartile Range(IQR) value.

#IQR is the distance between the 25th percentile point and the 50th percentile point.

#This method centres the median value at zero and this method is robust to outliers.

from scipy import stats 

IQR1 = stats.iqr(y1, interpolation = 'midpoint') 
y1_new = (y1-np.median(y1))/IQR1

IQR2 = stats.iqr(y2, interpolation = 'midpoint') 
y2_new = (y2-np.median(y2))/IQR2

plt.plot(x,y1_new,'red')
plt.plot(x,y2_new,'blue')


#%%
#%%
#Standardization
#**********************

#In standardization, we calculate the z-value for each of the data points 
#and replaces those with these values.

#This will make sure that all the features are centred around the mean value 
#with a standard deviation value of 1. 

#This is the best to use if your feature is normally distributed like 
#salary or age.

y1_new = (y1-np.mean(y1))/np.std(y1)
y2_new = (y2-np.mean(y2))/np.std(y2)

plt.plot(x,y1_new,'red')
plt.plot(x,y2_new,'blue')

#%%
#%%
#%%
#%%
#https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/

# spliting training and testing data
from sklearn.model_selection import train_test_split

X = df
y = target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=27)

#Normalization using sklearn
#--------------------------------
# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler

# fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# transform training data
X_train_norm = norm.transform(X_train)

# transform testing dataabs
X_test_norm = norm.transform(X_test)

#Standardization using sklearn
#----------------------------------
# data standardization with  sklearn
from sklearn.preprocessing import StandardScaler

# copy of datasets
X_train_stand = X_train.copy()
X_test_stand = X_test.copy()

# numerical features
num_cols = ['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']

# apply standardization on numerical features
for i in num_cols:
    
    # fit on training data column
    scale = StandardScaler().fit(X_train_stand[[i]])
    
    # transform the training data column
    X_train_stand[i] = scale.transform(X_train_stand[[i]])
    
    # transform the testing data column
    X_test_stand[i] = scale.transform(X_test_stand[[i]])

#You would have noticed that I only applied standardization to my numerical 
#columns and not the other One-Hot Encoded features. Standardizing the One-Hot 
#encoded features would mean assigning a distribution to categorical features. 
#You don’t want to do that!
    
#But why did I not do the same while normalizing the data? Because One-Hot 
#encoded features are already in the range between 0 to 1. 
#So, normalization would not affect their value.
    
#Comparing unscaled, normalized and standardized data

#K-Nearest Neighbours
#----------------------
# training a KNN model
from sklearn.neighbors import KNeighborsRegressor
# measuring RMSE score
from sklearn.metrics import mean_squared_error

# knn 
knn = KNeighborsRegressor(n_neighbors=7)

rmse = []

# raw, normalized and standardized training and testing data
trainX = [X_train, X_train_norm, X_train_stand]
testX = [X_test, X_test_norm, X_test_stand]

# model fitting and measuring RMSE
for i in range(len(trainX)):
    
    # fit
    knn.fit(trainX[i],y_train)
    # predict
    pred = knn.predict(testX[i])
    # RMSE
    rmse.append(np.sqrt(mean_squared_error(y_test,pred)))

# visualizing the result
df_knn = pd.DataFrame({'RMSE':rmse},index=['Original','Normalized','Standardized'])
df_knn

#%%
#Support Vector Regressor
#------------------------------
# training an SVR model
from  sklearn.svm import SVR
# measuring RMSE score
from sklearn.metrics import mean_squared_error

# SVR
svr = SVR(kernel='rbf',C=5)

rmse = []

# raw, normalized and standardized training and testing data
trainX = [X_train, X_train_norm, X_train_stand]
testX = [X_test, X_test_norm, X_test_stand]

# model fitting and measuring RMSE
for i in range(len(trainX)):
    
    # fit
    svr.fit(trainX[i],y_train)
    # predict
    pred = svr.predict(testX[i])
    # RMSE
    rmse.append(np.sqrt(mean_squared_error(y_test,pred)))

# visualizing the result    
df_svr = pd.DataFrame({'RMSE':rmse},index=['Original','Normalized','Standardized'])
df_svr

#%%
#Decision Tree
#--------------------
# training a Decision Tree model
from sklearn.tree import DecisionTreeRegressor
# measuring RMSE score
from sklearn.metrics import mean_squared_error

# Decision tree
dt = DecisionTreeRegressor(max_depth=10,random_state=27)

rmse = []

# raw, normalized and standardized training and testing data
trainX = [X_train,X_train_norm,X_train_stand]
testX = [X_test,X_test_norm,X_test_stand]

# model fitting and measuring RMSE
for i in range(len(trainX)):
    
    # fit
    dt.fit(trainX[i],y_train)
    # predict
    pred = dt.predict(testX[i])
    # RMSE
    rmse.append(np.sqrt(mean_squared_error(y_test,pred)))

# visualizing the result    
df_dt = pd.DataFrame({'RMSE':rmse},index=['Original','Normalized','Standardized'])
df_dt

#%%    























