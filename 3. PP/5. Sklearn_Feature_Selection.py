# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:38:47 2022

@author: rvamsikrishna
"""

#https://scikit-learn.org/stable/modules/feature_selection.html

#%%

#The classes in the sklearn.feature_selection module can be used for feature 
#selection/dimensionality reduction on sample sets, either to improve estimators’ 
#accuracy scores or to boost their performance on very high-dimensional datasets.

#%%
#1.13.1. Removing features with low variance **************************
#==============================================

#VarianceThreshold is a simple baseline approach to feature selection. 
#It removes all features whose variance doesn’t meet some threshold.
# By default, it removes all zero-variance features, i.e. features that have the same value in all samples.

#example, suppose that we have a dataset with boolean features, and we want to
# remove all features that are either one or zero (on or off) in more than 80% 
#of the samples. 

from sklearn.feature_selection import VarianceThreshold

X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)

#%%
#%%
#1.13.2. Univariate feature selection ***********************
#===========================================

#Univariate feature selection works by selecting the best features based on
# univariate statistical tests. 
#It can be seen as a preprocessing step to an estimator. 
#Scikit-learn exposes feature selection routines as objects that implement the transform method:



#1.SelectKBest removes all but the highest scoring features

#2.SelectPercentile removes all but a user-specified highest scoring percentage of features

#3.using common univariate statistical tests for each feature: false positive 
#rate SelectFpr, false discovery rate SelectFdr, or family wise error SelectFwe.

#4.GenericUnivariateSelect allows to perform univariate feature selection with a configurable strategy. This allows to select the best univariate selection strategy with hyper-parameter search estimator.



#These objects take as input a scoring function that returns univariate scores 
#and p-values (or only scores for SelectKBest and SelectPercentile):

#For regression: f_regression, mutual_info_regression
#For classification: chi2, f_classif, mutual_info_classif

#The methods based on F-test estimate the degree of linear dependency between 
#two random variables. 

#On the other hand, mutual information methods can capture any kind of statistical 
#dependency, but being nonparametric, they require more samples for accurate estimation.

#Feature selection with sparse data
#If you use sparse data (i.e. data represented as sparse matrices), chi2, 
#mutual_info_regression, mutual_info_classif will deal with the data without 
#making it dense.

#Warning : Beware not to use a regression scoring function with a 
#classification problem, you will get useless results. 

#%%
#1.SelectKBest removes all but the highest scoring features
#=============================================================
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X, y = load_iris(return_X_y=True)
X.shape

X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new.shape




#%%
#2.SelectPercentile removes all but a user-specified highest scoring percentage of features
#=========================
#Select features according to a percentile of the highest scores.

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2
X, y = load_digits(return_X_y=True)
X.shape
X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
X_new.shape

#%%
#sklearn.feature_selection.SelectFpr
#======================================

#Filter: Select the pvalues below alpha based on a FPR test.

#FPR test stands for False Positive Rate test. It controls the total amount 
#of false detections.

#sklearn.feature_selection.SelectFpr(score_func=<function f_classif>, *, alpha=0.05)

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFpr, chi2

X, y = load_breast_cancer(return_X_y=True)
X.shape

X_new = SelectFpr(chi2, alpha=0.01).fit_transform(X, y)
X_new.shape

#%%
# fit(X, y) - Run score function on (X, y) and get the appropriate features.
# fit_transform(X[, y]) - Fit to data, then transform it.
# get_feature_names_out([input_features]) - Mask feature names according to selected features.
# get_params([deep]) - Get parameters for this estimator.
# get_support([indices]) - Get a mask, or integer index, of the features selected.
# inverse_transform(X) - Reverse the transformation operation.
# set_params(**params) - Set the parameters of this estimator.
# transform(X) - Reduce X to the selected features.

#%%
#sklearn.feature_selection.SelectFdr
#==========================================

#Filter: Select the p-values for an estimated false discovery rate.

#This uses the Benjamini-Hochberg procedure. alpha is an upper bound on the
# expected false discovery rate.

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFdr, chi2
X, y = load_breast_cancer(return_X_y=True)
X.shape
X_new = SelectFdr(chi2, alpha=0.01).fit_transform(X, y)
X_new.shape

#%%
#%%
#%%
#1.13.3. Recursive feature elimination ******************
#========================================

#Given an external estimator that assigns weights to features (e.g., the coefficients 
#of a linear model), the goal of recursive feature elimination (RFE) is to select 
#features by recursively considering smaller and smaller sets of features. 

#First, the estimator is trained on the initial set of features and the importance 
#of each feature is obtained either through any specific attribute (such as coef_, 
#feature_importances_) or callable. 

#Then, the least important features are pruned from current set of features. 

#That procedure is recursively repeated on the pruned set until the desired 
#number of features to select is eventually reached.

#RFECV performs RFE in a cross-validation loop to find the optimal number of features.


from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)

estimator = SVR(kernel="linear")

selector = RFE(estimator, n_features_to_select=5, step=1)

selector = selector.fit(X, y)

selector.support_
selector.ranking_

#Methods :decision_function(X) --Compute the decision function of X.
#%%
#Recursive feature elimination with cross-validation to select the number of features.
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
import matplotlib.pyplot as plt

X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)

estimator = SVR(kernel="linear")

min_features_to_select = 1
selector = RFECV(estimator, step=1, cv=5,scoring="accuracy",min_features_to_select)

selector = selector.fit(X, y)

selector.support_
selector.ranking_
selector.n_features_

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(
    range(min_features_to_select, len(selector.grid_scores_) + min_features_to_select),
    selector.grid_scores_,
)
plt.show()

#%%
#1.13.4. Feature selection using SelectFromModel *********************
#=====================================================












