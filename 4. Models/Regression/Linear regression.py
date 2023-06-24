# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 05:47:53 2022

@author: rvamsikrishna
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

print(x)
print(y)

model = LinearRegression()
model.fit(x, y)

r_sq = model.score(x, y)

print('coefficient of determination:', r_sq)

print('intercept:', model.intercept_)

print('slope:', model.coef_)

#%%
new_model = LinearRegression().fit(x, y.reshape((-1, 1))
print('slope:', new_model.coef_)
print('intercept:', new_model.intercept_)

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')  

y_pred = model.intercept_ + model.coef_ * x
print('predicted response:', y_pred, sep='\n')
    
#%%
x_new = np.arange(5).reshape((-1, 1))
print(x_new)

y_new = model.predict(x_new)
print(y_new)

#%%
#Multiple Linear Regression With scikit-learn
#==============================================

import numpy as np
from sklearn.linear_model import LinearRegression

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

print(x)
print(y)

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
print('predicted response:', y_pred, sep='\n')

x_new = np.arange(10).reshape((-1, 2))
print(x_new)

y_new = model.predict(x_new)
print(y_new)

#%%
#Polynomial Regression With scikit-learn
#==========================================

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

transformer = PolynomialFeatures(degree=2, include_bias=False)

transformer.fit(x)

x_ = transformer.transform(x)

x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

print(x_)

model = LinearRegression().fit(x_, y)

r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)


x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
print(x_)


model = LinearRegression(fit_intercept=False).fit(x_, y)
r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)


y_pred = model.predict(x_)    
print('predicted response:', y_pred, sep='\n')


#%%
np.logspace(-6, 6, 13)









