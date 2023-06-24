# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:07:15 2022

@author: rvamsikrishna
"""
#https://www.kaggle.com/code/adhang/boston-house-prices-linear-regression

#https://www.kaggle.com/code/maxitype/2-linear-lasso-ridge-sgd-poly-beginner


#%%

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

sns.set_theme()
#%%

# Read Dataset
#-------------------

data = pd.read_csv('C:/Users/rvamsikrishna/Desktop/PY/Python/Models/Regression/boston.csv')
data.head()
#%%
data.info()
#%%
data.describe(include=['float']).T
#%%
data.describe(include=['int'])
#%%
print( 'unique value in RAD:', *data['RAD'].unique(), '\ncount:', 
      data['RAD'].nunique())

#%%
# check data type for each column
data.dtypes

#%%
# check total null values
data.isnull().sum()

#%%
# summarize the data type and null values just for better visual
data_type = pd.DataFrame(data.dtypes).T.rename({0:'Column Data Type'})

null_value = pd.DataFrame(data.isnull().sum()).T.rename({0:'Null Values Count'})

# combine the data
data_info = data_type.append(null_value)
data_info

#%%
# our data has no null values, but I want to see the percentage
null_percentage = pd.DataFrame(data.isnull().sum()/data.shape[0]*100).T.rename({0:'Percentage of Null Values'}).round(2)

# combine the data
data_info = data_info.append(null_percentage)
data_info

#%%
# Exploratory Data Analysis
#--------------------------------

# descriptive statistics
data.describe().round(2)

#%%

#Data Distribution
#------------------
#We can use the histogram to see the data distribution. 
#There are several ways to make a histogram (and others) plot.

sns.histplot(data=data, x='RM', bins=20)
plt.show()

#The distribution of RM column is good because it's not skewed. How about the other?

#%%
#Histogram of Data Distribution¶
#====================================

#A histogram plot is a discrete plot, with some bins to represent 
#the distribution of data.

# get all column name
column_list = list(data.columns)

fig, ax = plt.subplots(5,3, figsize=(12,10), constrained_layout=True)

# axes_list = []
# for axes_row in ax:
#     for axes in axes_row:
#         axes_list.append(axes)


# it's the same as the above looping, but in a shorter way
# it may be difficult to understand for a beginner
axes_list = [axes for axes_row in ax for axes in axes_row]

for i, col_name in enumerate(column_list):
    sns.histplot(data=data, x=col_name, ax=axes_list[i], bins=20)

# I will hide the last axes since it's an empty plot
axes_list[14].set_visible(False)

# remove axes line on top-right
sns.despine()
plt.show()


#%%
#Density of Data Distribution
#==============================

#A density plot is a continuous (smoothed) version of a histogram. 
#Our data is not continuous, there may be some gap between values, 
#So, we need an estimator. The most common form of estimation is 
#kernel density estimation (KDE).

fig, ax = plt.subplots(5,3, figsize=(12,10), constrained_layout=True)

axes_list = [axes for axes_row in ax for axes in axes_row]

for i, col_name in enumerate(column_list):
    sns.kdeplot(data=data, x=col_name, ax=axes_list[i])

# I will hide the last axes since it's an empty plot
axes_list[14].set_visible(False)

# remove axes line on top-right
sns.despine()
plt.show()

#%%

# Histogram and Density--We can combine histogram and density in 1 plot.
#=======================

fig, ax = plt.subplots(5,3, figsize=(12,10), constrained_layout=True)

axes_list = [axes for axes_row in ax for axes in axes_row]

for i, col_name in enumerate(column_list):
    sns.histplot(data=data, x=col_name, ax=axes_list[i], bins=20, 
                 kde=True, line_kws={'linewidth':3})

# I will hide the last axes since it's an empty plot
axes_list[14].set_visible(False)

# remove axes line on top-right
sns.despine()
plt.show()

#%%

#As you can see,some attributes are skewed and not in normal distribution form.

#Handling Skewed Data

#Some methods for handling skewed data:
#Log transform
#Square root transform
#Box-cox transform

data.skew()

#%%
# combine with our data previous data info
data_skewness = pd.DataFrame(data.skew()).T.rename({0:'Data Skewness'}).round(2)

data_info = data_info.append(data_skewness)
data_info

#%%
#We can see thatCRIM has the largest skewness. Let's fix it.

#Log Transform--We can use Numpy log function to transform our data.

crim_log = np.log(data['CRIM'])

crim_log.skew().round(2)

#%%
# Using log transform, we have reduced the CRIM skewness. 
#But, how about the distribution of CRIM data? Let's visualize it

sns.histplot(crim_log, bins=20, kde=True, line_kws={'linewidth':3})
sns.despine()
plt.show()

#It's still not in normal distribution form, but it's way much 
#better than before.

#%%
#Square Root Transform -Again, we can use Numpy function to transform our data.
#======================

crim_sqrt = np.sqrt(data['CRIM'])
crim_sqrt.skew().round(2)

#%%
fig, ax = plt.subplots(1, 2, figsize=(12,4), constrained_layout=True)

sns.histplot(data['CRIM'], bins=20, kde=True, ax=ax[0], line_kws={'linewidth':3})
ax[0].set_ylim(0,400)

sns.histplot(crim_sqrt, bins=20, kde=True, ax=ax[1], line_kws={'linewidth':3})
ax[1].set_ylim(0,400)

sns.despine()
plt.show()

#skewness From 5.22 to 2.02, I think it's not a really big difference.
# And the distribution is almost the same, but the range is smaller.

#%%
#Box-Cox Transform
#===================

#there's one thing that you have to pay attention to when using this 
#transformation, your data must be positive.

#We can use Scipy library for this transformation.

from scipy import stats

cal_crim_boxcox = stats.boxcox(data['CRIM'])[0]
# cal_crim_boxcox

#Since it returns an array, we need to convert it to Series 
#(or Dataframe) to see the skewness.

crim_boxcox = pd.Series(cal_crim_boxcox)
crim_boxcox.skew().round(2)

#%%
#Great, now our skewness is really small. Let's see the data distribution again.

sns.histplot(crim_boxcox, bins=20, kde=True, line_kws={'linewidth':3})
sns.despine()
plt.show()

#It has similar distribution with the log transform. 
#But I think it's better since the skewness is lower than using log transform.

#%%
#Notes: For this project, I won't use CRIM attribute just for simplicity 
#reasons. I do skewed data handling just for learning purposes. To be 
#honest, I'm not using data that had been transformed because I still 
#don't know how to reverse transform. So, maybe I'll do it in the next project.

#%%
#Correlation
#===============
#It's used to see the relationship between features
#Correlation value is between -1 to 1
#Correlation value = -1 means negative correlation. If 'X' goes bigger, 'Y' goes smaller
#Correlation value = 1 means positive correlation. If 'X' goes bigger, 'Y' also goes bigger
#Correlation value = 0 means, there's no correlation between 'X' and 'Y'

corr_matrix = data.corr().round(2)
corr_matrix

#%%
#Visualization can help us to see correlation more clearly.

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, center=0, vmin=-1, vmax=1, mask=mask, annot=True, cmap='BrBG')
plt.show()

#%%
#Based on the above correlation plot, we can see that RM and LSTAT have
# the highest correlation with MEDV. We can use a scatter plot to see 
#the correlation between attributes.

# RM and MEDV

plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='RM', y='MEDV')
sns.despine()
plt.show()

#%%
#From this plot, we can see that RM has a positive correlation with MEDV, 
#just like the correlation heatmap above.

# LSTAT and MEDV

plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='LSTAT', y='MEDV')
sns.despine()
plt.show()

#From this plot, we can see that LSTAT has a negative correlation 
#with MEDV, just like the correlation heatmap above.

#***************************************************************************
#***************************************************************************
#%%
#Univariate Linear Regression
#===============================

#Feature Selection

#Since RM has the highest correlation to MEDV, I will use this 
#attribute to make univariate linear regression.

# I use [[]] to create a dataframe
# if you use [], it will create a series

X = data[['RM']]
X.head()

Y = data[['MEDV']]
Y.head()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=1)

# let's see the shape of each dataset

print(X.shape)
print(Y.shape)
print()
print(X_train.shape)
print(Y_train.shape)
print()
print(X_test.shape)
print(Y_test.shape)

# instantiating the model
model = LinearRegression()

model.fit(X_train, Y_train)

# first, let's see the coefficient value (a)
model_coef = model.coef_
model_coef.round(2)

# model intercept (b)
model_intercept = model.intercept_
model_intercept.round(2)


# predict test dataset
y_test_pred = model.predict(X_test)

# let's check the prediction and the actual value
print(Y_test[:5].values)
print()
print(y_test_pred[:5].round(2))

#%%
#Model Evaluation
#--------------------
plt.scatter(X_test, Y_test, label='test data', color='k')
plt.plot(X_test, y_test_pred, label='pred data', color='b', linewidth=3)
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.title('Model Evaluation')
plt.legend(loc='upper left')

# plt.savefig('./output/linear-regression.png')
plt.show()

#%%
residual = Y_test - y_test_pred

plt.scatter(X_test, residual, color='k')
plt.hlines(y=0, xmin=X_test.min(), xmax=X_test.max(), linestyle='--', linewidth=3)
plt.xlim((4,9))
plt.xlabel('RM')
plt.ylabel('residual')
plt.title('Residual')
plt.show()

#%%
#Mean Squared Error
#------------------------
mean_squared_error(Y_test, y_test_pred).round(2)

#In general, the smaller the MSE, the better, yet there is no absolute good or bad threshold.
#We can define it based on the dependent variable, i.e., MEDV in the test set.
#To make the scale of errors to be the same as the scale of targets, 
#root mean squared error (RMSE) is often used. It is the square root of MSE.

np.sqrt(mean_squared_error(Y_test, y_test_pred)).round(2)


#%%
#R-Squared
#-------------

#It is the proportion of total variation explained by the model. 
#We can use score() to obtain the R-squared value.

model.score(X_test, Y_test).round(3)


#************************************************
#If you want to manually calculate R-squared, you need to calculate 
#the variance of the test dataset and residual.

#Calculate the difference between each datapoint with the average
#Calculate the squared of difference
#Sum all of the squared roots, it's called variance

# step 1, calculate the difference
diff = (Y_test - Y_test.mean())
diff.head()

# step 2, calculate the squared of difference
squared = diff**2
squared.head()



# step 3, sum all the squared root
variance_test = squared.sum()
variance_test.round(2)


#Total Variance of Residual
#Since residual = Y_test - y_test_pred, we can say residual = diff
#I will calculate the variance in 1 line of code to make it simple

variance_residual = (residual**2).sum()
variance_residual.round(2)


#Proportion of Total Variation---It's R-squared.
R_sqr = 1 - (variance_residual)/(variance_test)
R_sqr.round(3)

#%%
#**************************************************************
#Multivariate Linear Regression¶
#===================================
# feature selection
X2 = data[['RM','LSTAT']]
Y = data[['MEDV']]

# train - test split
X2_train, X2_test, Y_train, Y_test = train_test_split(X2, Y, test_size=0.3, random_state=1)

# instantiating the model
model_2 = LinearRegression()

# fitting the model
model_2.fit(X2_train, Y_train)

model_2_coef = model_2.coef_.round(2)
model_2_coef

model_2_intercept = model_2.intercept_.round(2)
model_2_intercept

# predict
y_test_pred_2 = model_2.predict(X2_test)

#%%
#Visualizing the Data

# I'm resetting the seaborn theme because the surface can't be transparent
sns.reset_orig()

# visualize the data in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# X_viz = X2.values.reshape(-1,2)
# x1 = X_viz[:,0]
# x2 = X_viz[:,1]
x1 = X2['RM']
x2 = X2['LSTAT']
x3 = Y['MEDV']

ax.scatter3D(x1, x2, x3, c=x3, cmap='viridis');

plt.show()

#%%
# adding a meshgrid
x_surf, y_surf = np.meshgrid(np.linspace(x1.min(), x1.max(), 100), 
                             np.linspace(x2.min(), x2.max(), 100))

onlyX = pd.DataFrame({
        'RM':x_surf.ravel(),
        'LSTAT':y_surf.ravel()})
fittedY = model_2.predict(onlyX)
fittedY = np.array(fittedY)

fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, x3, c=x3, cmap='viridis', marker='o', alpha=0.5)
ax.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape), color='None', alpha=0.3)
ax.set_xlabel('RM')
ax.set_ylabel('LSTAT')
ax.set_zlabel('MEDV')
ax.view_init(elev=15, azim=30)
plt.show()

#%%
# plot with different point of view
fig = plt.figure(figsize=(12,4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')
axes = [ax1, ax2, ax3]

for ax in axes:
    ax.scatter(x1, x2, x3, c=x3, cmap='viridis', marker='o', alpha=0.5)
    ax.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape), color='None', alpha=0.3)
    ax.set_xlabel('RM')
    ax.set_ylabel('LSTAT')
    ax.set_zlabel('MEDV')

ax1.view_init(elev=15, azim=30)
ax2.view_init(elev=15, azim=-120)
ax3.view_init(elev=30, azim=45)
fig.tight_layout()
plt.show()

#%%
#Comparing Model¶
#-----------------
mse_1 = mean_squared_error(Y_test, y_test_pred)
mse_2 = mean_squared_error(Y_test, y_test_pred_2)

print('Univariate Linear Regression MSE:', mse_1.round(2))
print('Multivariate Linear Regression MSE:', mse_2.round(2))

# calculate improvement of MSE
mse_percentage = (mse_1-mse_2)/mse_1 * 100

print(f'Improvement: {mse_percentage.round(2)}% reduction')

#%%
#R-Squared¶
#-------------
R_sqr_1 = model.score(X_test, Y_test)
R_sqr_2 = model_2.score(X2_test, Y_test)

print('Univariate Linear Regression R-Squared:', R_sqr_1.round(3))
print('Multivariate Linear Regression R-Squared:', R_sqr_2.round(3))

# calculate improvement of MSE
R_sqr_percentage = (R_sqr_2-R_sqr_1)/R_sqr_1 * 100

print(f'Improvement: {R_sqr_percentage.round(2)}% adjustment')

#%%
#%%
#%%
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures



X = data.drop(['MEDV'],axis=1)
y = data['MEDV']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve

def print_metrics(y_test, pred):  
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    RMSE = np.sqrt(mse).round(2)
    r2 = r2_score(y_test, pred)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', RMSE)
    print('R2:', r2)
    print('')
    
def save_metrics(y_test,pred):
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    RMSE = np.sqrt(mse).round(2)
    r2 = r2_score(y_test, pred)
    return mae, mse, RMSE, r2    

#BaseLine for Linear Regression
#=================================
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_pred_train = lr.predict(X_train)

print('metrics for test:\n')
print_metrics(y_test, lr_pred)
print('metrics for train:\n')
print_metrics(y_train, lr_pred_train)

lr_mae, lr_mse, lr_rmse, lr_r2 = save_metrics(y_test, lr_pred)

lr_feature = pd.DataFrame(lr.coef_, index=X.columns, 
                          columns=['coef']).sort_values(['coef'], ascending=False)

lr_feature

#AGE and ZN have the smallest coefficient

lr_feature.plot(kind='barh', figsize=(7,7));

sns.scatterplot(x=y_test, y=lr_pred);


#Learning curve. Determines cross-validated training and test scores for different 
#training set sizes. A cross-validation generator splits the whole dataset k times
# in training and test data.

#Use learning_curve() to generate the data needed to plot a learning curve.
# The function returns a tuple containing three elements: the training set sizes,
# and the error scores on both the validation sets and the training sets

train_sizes=[1,101,203,304,404]

train_sizes, train_scores, validation_scores = learning_curve(
        estimator = LinearRegression(),
        X = X,
        y = y, 
        train_sizes = train_sizes, 
        cv = 5, 
        shuffle=True,
        scoring = 'neg_mean_squared_error',random_state=17)


train_scores_mean = -train_scores.mean(axis = 1)
validation_scores_mean = -validation_scores.mean(axis =1)
print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))


plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MSE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,40)
plt.xlim(0,404)

#%%
#%%
#SGDRegressor¶
#===================
sgd = SGDRegressor(random_state=0, tol=.0001)

sgd_params = {'alpha':[.001, .0015, .01, .015, .1, 1, 10, 100],
             'eta0':[.001, .003, .01, .03, .1, .3, 1, 3]
             }    


gs_sgd = GridSearchCV(sgd, sgd_params, cv=5, n_jobs=-1, 
                      verbose=1, scoring='neg_mean_squared_error')

gs_sgd.fit(X_train, y_train)


sgd_best = gs_sgd.best_estimator_

sgd_pred = sgd_best.predict(X_test)
sgd_pred_train = sgd_best.predict(X_train)

print('metrics for test:\n')
print_metrics(y_test, sgd_pred)
print('metrics for train:\n')
print_metrics(y_train, sgd_pred_train)

sgd_mae, sgd_mse, sgd_rmse, sgd_r2 = save_metrics(y_test, sgd_pred)

#%%
#Ridge Regression¶
#=======================

r_lr = RidgeCV(alphas=[.001, .0015, .01, .015, .1, 1, 10, 100], cv=5)
r_lr.fit(X_train, y_train)


r_lr_pred = r_lr.predict(X_test)
r_lr_pred_train = r_lr.predict(X_train)


print('metrics for test:\n')
print_metrics(y_test, r_lr_pred)
print('metrics for train:\n')
print_metrics(y_train, r_lr_pred_train)

r_lr_mae, r_lr_mse,r_lr_rmse, r_lr_r2 = save_metrics(y_test, r_lr_pred)


#%%
#Lasso Regression¶
#=====================
l_lr = LassoCV(alphas=[.001, .0015, .01, .015, .1, 1, 10, 100], random_state=0, cv=5)
l_lr.fit(X_train, y_train)


l_lr_pred = l_lr.predict(X_test)
l_lr_pred_train = l_lr.predict(X_train)


print('metrics for test:\n')
print_metrics(y_test, l_lr_pred)
print('metrics for train:\n')
print_metrics(y_train, l_lr_pred_train)

l_lr_mae, l_lr_mse, l_lr_rmse, l_lr_r2 = save_metrics(y_test, r_lr_pred)


#%%
#Elastic Net Regression¶
#==========================
el_lr = ElasticNetCV(alphas=[.001, .0015, .01, .015, .1, 1, 10, 100],
                     l1_ratio=[.1, .3, .5, .7, .9, 1],
                     random_state=0, cv=5)
el_lr.fit(X_train, y_train)


el_lr_pred = el_lr.predict(X_test)
el_lr_pred_trian = el_lr.predict(X_train)


print('metrics for test:\n')
print_metrics(y_test, el_lr_pred)
print('metrics for train:\n')
print_metrics(y_train, el_lr_pred_trian)

el_lr_mae, el_lr_mse, el_lr_rmse, el_lr_r2 = save_metrics(y_test, l_lr_pred)


#%%
#Polynominal Regression¶
#==========================
pol_lr = PolynomialFeatures(degree=2)

X_train_pol_2 = pol_lr.fit_transform(X_train)
X_test_pol_2 = pol_lr.transform(X_test)


lr.fit(X_train_pol_2, y_train)
pol_pred = lr.predict(X_test_pol_2)
pol_pred_train =lr.predict(X_train_pol_2)


print('metrics for test:\n')
print_metrics(y_test, pol_pred)
print('metrics for train:\n')
print_metrics(y_train, pol_pred_train)


pol_mae, pol_mse, pol_rmse, pol_r2 = save_metrics(y_test, pol_pred)


#%%

algoritm = ['Linear Regression','SGD Regression','Ridge Regression',
            'Lasso Regression','Elastic Net Regression','Polynominal Regression']

result = {'MAE':[lr_mae, sgd_mae, r_lr_mae, l_lr_mae, el_lr_mae, pol_mae],
         'MSE':[lr_mse, sgd_mse, r_lr_mse, l_lr_mse, el_lr_mse, pol_mse],
         'RMSE':[lr_rmse, sgd_rmse, r_lr_rmse, l_lr_rmse, el_lr_rmse, pol_rmse],
         'R2':[lr_r2, sgd_r2, r_lr_r2, l_lr_r2, el_lr_r2, pol_r2]
         }

Result = pd.DataFrame(result, index=algoritm)
Result



#%%




