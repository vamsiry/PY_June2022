# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:58:54 2022

@author: vamsi
"""

#%%
#Understanding the data
#----------------------------
# Important Imports
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib as mpl
import matplotlib.pyplot as plt

# Load the data
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)


train_df = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
train_df.head()

#%%
# Now I just want to see how many categorical features I have and how many
#   categories per feature
# I also want to know unique numbers in each numerical features since 
#   it could be categorical as well

import  pandas as pd

def display_information(df):
    info = pd.DataFrame(df.dtypes, columns=['dtypes'])
    info = info.reset_index()
    info['Name'] = info['index']
    info = info[['Name', 'dtypes']]
    info['Uniques'] = df.nunique(dropna=False).values
    info['Missing'] = df.isnull().sum().values
    
    return info

display_information(train_df)        


#%%
# Looking at this table you can notice that you have 1460 rows,
# feature Alley has 1369 missing values
# feature PoolQC has 1453 missing values
# feature Fence has 1179 missing values
# feature MiscFeature has 1406 missing values
# these amounts of missing values are too big to be imputed, so we will drop these features

train_df.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature"], inplace=True)

#%%
# now let's plot some of the Categorical data
def plot_categorical_count(df):
    color = sb.color_palette()[0]
    columns = df.select_dtypes(["object"]).columns        
    
    fig, ax = plt.subplots(len(columns) // 4, 4, figsize=(22, len(columns)))
    for col, subplot in zip(columns, ax.flatten()):
        freq = df[col].value_counts()
        sb.countplot(df[col], order=freq.index, ax=subplot, color=color)
        
        
plot_categorical_count(train_df)

#%%
# the following function is going to allow us to somehow see the relation between
# categorical features and the output
def plot_categorical_relation_to_target(df, target):
    color = sb.color_palette()[0]
    columns = df.select_dtypes(["object"]).columns
        
    fig, ax = plt.subplots(len(columns) // 4, 4, figsize=(22, len(columns)))
    for col, subplot in zip(columns, ax.flatten()):
        freq = df[col].value_counts()
        sb.violinplot(data=df, x=col, y=target, order=freq.index, ax=subplot, color=color)
        
plot_categorical_relation_to_target(train_df, "SalePrice")

#%%
#What I figured
#By looking at these two figures I can tell that encoding the Utilities 
#and Street features is useless since most of the data exist in one 
#category and the other category has price that is in the mean of the 
#first category so it seems not to add any value.

train_df.drop(columns=["Utilities", "Street"], inplace=True)

#%%
# plot count and violin of feature
# so we can see the frequency of categories in a certain feature and
# see how it relates to the output target
def count_and_violin(df, feature, target):
    color = sb.color_palette()[0]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    axis = ax.flatten()
    freq = df[feature].value_counts()
    sb.countplot(df[feature], order=freq.index, ax=axis[0], color=color)
    sb.violinplot(data=df, x=feature, y=target, order=freq.index, ax=axis[1], color=color)
    plt.show()
    
count_and_violin(train_df, "MSZoning", "SalePrice")

#%%
train_df.describe()

#%%
corr = train_df.corr()
corr.style.background_gradient(cmap='coolwarm')

#What We notice
#------------------
#Ideally you would want features that correlate with the target column if 
#you are going for linear regression, in that case overallQuall seems to be
# very linearly correlated, which makes sense. however mssubclass is not 
#linearly correlated, which means it is not useful for linear models, 
#we will keep them for now and use recursive feature elimination or 
#pca later, but only after encoding.

#One more thing you need to know is that features that correlate with each 
#other are not that useful, like YearBuilt, and GarageYrBlt since most 
#likely they are going to be the same, we only need one of them, and
# this is visually represented in the correlation matrix.



#%%
# mapping some ordinal features in the correct order
# because the OrdinalEncoder uses alphabitical order which is not meaningful
# you could override the default behaviour of the OrdinalEncoder by passing a list of
# categories in the correct order but then you would need to create an encoder for each feature
ordinal_maps = {
    "LandSlope": {'Gtl': 3, 'Mod': 2, 'Sev': 1},
    "ExterQual": {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    "ExterCond":{'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    "BsmtQual": {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    "BsmtCond": {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    "BsmtExposure": {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0},
    "Functional": {'Typ': 6, 'NA': 6, 'Min1': 5, 'Min2': 4, 'Mod': 3, 'Maj1': 2, 'Maj2': 1, 'Sev': 0}
}

def preprocess_ordered_ordinals(df, maps):
    for key in maps:
        df[key] = df[key].map(maps[key])
        
preprocess_ordered_ordinals(train_df, ordinal_maps)


#%%
#Preprocessing and feature engineering
#----------------------------------------
# Here are most of the useful features split on three lists according to there types
# feel free to drop some of them that you think might not be useful.
# Read the data_description.txt file carefully and reason about the data to elemenate some of the features.
# also use the correlation matrix and the count_and_violin function to see how the feature affects the target.
# MSSubClass is still a categorical feature even though it is of type int

nominal = ["MSZoning", "LotShape", "LandContour", "LotConfig", "Neighborhood",
           "Condition1", "BldgType", "RoofStyle",
           "Foundation", "CentralAir", "SaleType", "SaleCondition"]

ordinal = ["LandSlope", "OverallQual", "OverallCond", "YearRemodAdd",
          "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
          "KitchenQual", "Functional", "GarageCond", "PavedDrive"]

numerical = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtUnfSF",
            "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea",
            "OpenPorchSF"]


#%%
#Let's create some pipelines
#-----------------------------
#Now we are going to use the lists above to preprocess each feature 
#according to its type.

# We are going to create pipelines for the ease of iteration and model evaluation.

### Importing all needed modules
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from yellowbrick.model_selection import RFECV

#%%
# Choosing only the useful features
X = train_df[nominal + ordinal + numerical]
y = train_df["SalePrice"]


#%%
nominal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse=True, handle_unknown="ignore"))
])


#%%    
ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", preprocess_ordered_ordinals(train_df, ordinal_maps)
])


#%%    
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])


#%%   
#Now let's join all of the above in one pipeline that targets each column with
# its family's pipeline.

# We can do so using the sklearn.compose.ColumnTransformer Object

from sklearn.compose import ColumnTransformer

# here we are going to instantiate a ColumnTransformer object with a list of tuples
# each of which has a the name of the preprocessor
# the transformation pipeline (could be a transformer)
# and the list of column names we wish to transform
    
    
basic_preprocessor = ColumnTransformer([
    ("nominal_preprocessor", nominal_pipeline, nominal),
    ("ordinal_preprocessor", ordinal_pipeline, ordinal),
    ("numerical_preprocessor", numerical_pipeline, numerical),
])

#X_preprocessed = basic_preprocessor.fit_transform(X)

#%%
from sklearn.linear_model import LinearRegression

complete_pipeline = Pipeline([
    ("preprocessor", basic_preprocessor),
    ("estimator", LinearRegression())
])

complete_pipeline.fit(X, y=y)

score = complete_pipeline.score(X, y)
print(score)

complete_pipeline.predict(X).shape


#%%
#%%
#Model selection¶
#----------------------
#Now since this is a regression problem with many features, and high 
#linear correlation between those features and the target, we can use 
#linear regression.

# We will also try ensemble learning and see if it gives better results.

#In the following cells, you will learn:

#1.How to train a scikit-learn estimator (model)
#2.How to evaluate the model using cross validation

#%%
from sklearn.linear_model import LinearRegression
# model definition
model = LinearRegression()
# scores is an array of five elements each element is the scoring on a certain fold
scores = cross_val_score(model, X_preprocessed, y, scoring="neg_root_mean_squared_error", cv=3)
# we are going to use the mean of these scores as an evaluation metric
print(scores.mean())

#%%
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=500)
scores = cross_val_score(model, X_preprocessed, y, scoring="neg_root_mean_squared_error", cv=5)
print(scores.mean())

#%%
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=500)
scores = cross_val_score(model, X_preprocessed, y, scoring="neg_root_mean_squared_error", cv=5)
print(scores.mean())

#%%
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=500)
scores = cross_val_score(model, X_preprocessed, y, scoring="neg_root_mean_squared_error", cv=5)
print(scores.mean())

#%%
#Grid search¶
#-----------------
#It look like GradientBoostingRegressor gave the best results. Now let's 
#figure out which parameters we should give to this model to get the best results.

# There are two methods:
#1.Grid Search
#2.Random Search In the first you try the model with all possible 
#combinations of given parameters and choose the best one, the latter 
#doesn't look at all combinations, it only look at random samples which
# makes it more effecient but less optimal.

#%%
# Since we don't care that much about time since the model is fairly small,
# we are going to use grid search

from sklearn.model_selection import GridSearchCV

## First you need to define the parameter space, or the grid
# it is a simple dictionary where:
# the keys are the names of the parameters given to the estimator
# the values are lists of possible values for each parameter
# the search is going to go as follows
# {"n_estimators": 100, learning_rate: 0.001, max_depth: 3}
# {"n_estimators": 100, learning_rate: 0.001, max_depth: 5}
# {"n_estimators": 100, learning_rate: 0.001, max_depth: 7}
# {"n_estimators": 100, learning_rate: 0.005, max_depth: 3}
# {"n_estimators": 100, learning_rate: 0.005, max_depth: 5}
# {"n_estimators": 100, learning_rate: 0.005, max_depth: 7}
# ........

grid = {
    "n_estimators": range(300, 500, 100),
    "learning_rate": [0.05, 0.1, 0.5],
    "max_depth": [2, 3, 4],
}

searcher = GridSearchCV(GradientBoostingRegressor(), grid, scoring="neg_root_mean_squared_error",
                        n_jobs=-1, cv=3, return_train_score=True)

searcher.fit(X_preprocessed, y)

searcher.best_params_

#%%
# the return_train_score=True paramter allows us to look at the results of the whole process
scores = pd.DataFrame(searcher.cv_results_)
scores

#%%
# the best validation score
searcher.best_score_

#%%
# the parameters that got the best score
searcher.best_params_

#%%
# the trained model that got the best score
model = searcher.best_estimator_

#%%
#Predictions¶
#------------
#Now that we have a pretty good pipeline and a pretty good model
# let's submit our predictions
test_df = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
test_df.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature", "Utilities", "Street"], inplace=True)

#%%
preprocess_ordered_ordinals(test_df, ordinal_maps)

#%%
X = test_df[nominal + ordinal + numerical]

#%%
X_preprocessed = basic_preprocessor.transform(X)

#%%
preds = model.predict(X_preprocessed)

#%%
submission = pd.read_csv("/kaggle/input/home-data-for-ml-course/sample_submission.csv")
submission.head()

#%%
submission['SalePrice'] = preds
submission.head()

#%%
submission.to_csv("submission.csv", index=False)

#%%
pd.read_csv("submission.csv").head()

















