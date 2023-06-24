# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:47:05 2022

@author: rvamsikrishna
"""

#https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/

#https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/
#https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/

#An Outlier is a data-item/object that deviates significantly from the rest of the data
#They can be caused by measurement or execution errors. 
#The analysis for outlier detection is referred to as outlier mining.\

# How to treat outliers?
#--------------------------
#1. Trimming
#2. Capping

# Importing
import sklearn
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
bos_hou = load_boston()

# Create the dataframe
column_name = bos_hou.feature_names
df_boston = pd.DataFrame(bos_hou.data)
df_boston.columns = column_name
df_boston.head()

#%%
#Outliers can be detected using visualization, implementing mathematical formulas 
#on the dataset, or using the statistical approach. All of these are discussed below. 

#Example 1: Using Box Plot
#--------------------------
#Boxplot summarizes sample data using 25th, 50th, and 75th percentiles.
# One can just get insights(quartiles, median, and outliers) into the dataset
# by just looking at its boxplot.

# Box Plot
import seaborn as sns
sns.boxplot(df_boston['DIS'])

# Position of the Outlier
print(np.where(df_boston['DIS']>10))

#%%
#Example 2: Using ScatterPlot.
#----------------------------------
#It is used when you have paired numerical data, or when your dependent variable
# has multiple values for each reading independent variable, or when trying to
# determine the relationship between the two variables. In the process of
# utilizing the scatter plot, one can also use it for outlier detection.

import matplotlib.pyplot as plt

# Scatter plot
fig, ax = plt.subplots(figsize = (18,10))

ax.scatter(df_boston['INDUS'], df_boston['TAX'])

# x-axis label
ax.set_xlabel('(Proportion non-retail business acres)/(town)')

# y-axis label
ax.set_ylabel('(Full-value property-tax rate)/( $10,000)')
plt.show()


# Position of the Outlier
print(np.where((df_boston['INDUS']>20) & (df_boston['TAX']>600)))


#%%
#2. Z-score
#------------------
#Z- Score is also called a standard score. This value/score helps to understand 
#that how far is the data point from the mean. And after setting up a threshold 
#value one can utilize z score values of data points to define the outliers.

#Zscore = (data_point -mean) / std. deviation

# Z score
from scipy import stats
import numpy as np

z = np.abs(stats.zscore(df_boston['DIS']))
#print(z)


threshold = 3

# Position of the outlier
print(np.where(z > 3))

#%%
import numpy as np

outliers = []

def detect_outliers_zscore(data):
    thres = 3
    mean = np.mean(data)
    std = np.std(data)
    # print(mean, std)
    for i in data:
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            outliers.append(i)
            return outliers# Driver code


sample_outliers = detect_outliers_zscore(sample)
print("Outliers from Z-scores method: ", sample_outliers)

#%%
#3. IQR (Inter Quartile Range)
#----------------------------------
#IQR (Inter Quartile Range) is used to finding the outliers is the most 
#commonly used and most trusted approach used in the research field.

# IQR
Q1 = np.percentile(df_boston['DIS'], 25,
				interpolation = 'midpoint')

Q3 = np.percentile(df_boston['DIS'], 75,
				interpolation = 'midpoint')
IQR = Q3 - Q1

# Above Upper bound
upper = df_boston['DIS'] >= (Q3+1.5*IQR)
#print("Upper bound:",upper)
print(np.where(upper))


# Below Lower bound
lower = df_boston['DIS'] <= (Q1-1.5*IQR)
#print("Lower bound:", lower)
print(np.where(lower))

#%%
#***************************************************************************
#**************Removing/deleting the outliers Using IQR***********************
#****************************************************************************

# Importing
import sklearn
from sklearn.datasets import load_boston
import pandas as pd

# Load the dataset
bos_hou = load_boston()

# Create the dataframe
column_name = bos_hou.feature_names
df_boston = pd.DataFrame(bos_hou.data)
df_boston.columns = column_name
df_boston.head()

''' Detection '''
# IQR
Q1 = np.percentile(df_boston['DIS'], 25,
				interpolation = 'midpoint')

Q3 = np.percentile(df_boston['DIS'], 75,
				interpolation = 'midpoint')
IQR = Q3 - Q1

print("Old Shape: ", df_boston.shape)

# Upper bound
upper = np.where(df_boston['DIS'] >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(df_boston['DIS'] <= (Q1-1.5*IQR))

''' Removing the Outliers '''
df_boston.drop(upper[0], inplace = True)
df_boston.drop(lower[0], inplace = True)

print("New Shape: ", df_boston.shape)


#%%
#*************************************************************************
#****************** Percintile capping ******************************
#*************************************************************************
#The data points that are lesser than the 10th percentile are replaced with the
# 10th percentile value and the data points that are greater than the 
#90th percentile are replaced with 90th percentile value.

tenth_percentile = np.percentile(sample, 10)

ninetieth_percentile = np.percentile(sample, 90)

# print(tenth_percentile, ninetieth_percentile)

b = np.where(sample<tenth_percentile, tenth_percentile, sample)
b = np.where(b>ninetieth_percentile, ninetieth_percentile, b)
# print("Sample:", sample)
print("New array:",b)

#%%
for col in df.columns:
    percentiles = df[col].quantile([0.01, 0.99]).values
    df[col][df[col] <= percentiles[0]] = percentiles[0]
    df[col][df[col] >= percentiles[1]] = percentiles[1]
    
#%%
#numpy.clip : Clip (limit) the values in an array.For example, if an interval 
#of [0, 1] is specified, values smaller than 0 become 0, and values larger
# than 1 become 1.
    
import numpy as np
for col in df.columns:
    percentiles = df[col].quantile([0.01, 0.99]).values
    df[col] = np.clip(df[col], percentiles[0], percentiles[1])

#%%    







