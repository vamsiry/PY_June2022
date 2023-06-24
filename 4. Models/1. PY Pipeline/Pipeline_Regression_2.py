# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:35:42 2022

@author: vamsi
"""

#https://dockship.io/articles/60128f3c5c9276402bd7a145/a-comprehensive-guide-for-scikit-learn-pipelines

#Why Sklearn Pipeline?
#1. Combine the preprocessing step with the inference step at one object.
#2. Save the complete pipeline to disk.
#3. Easily experiment with different techniques of preprocessing.
#4. Pipeline reuse.
#5. Easy cloud deployment.

#Datalink : https://www.kaggle.com/c/home-data-for-ml-course/data?select=test.csv

#House price prediction complete EDA
#https://www.kaggle.com/code/mahmoud1youssef/housing-prices-full-solution/notebook

#%%
#How Sklearn Pipeline?

import pandas as pd
import numpy as np

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

## let's create a validation set from the training set

msk = np.random.rand(len(train_df)) < 0.8

val_df = train_df[~msk]

train_df = train_df[msk]


#%%
#Feature selection
#--------------------
#This data has 163 columns, however, we are not going to use all of them.
#After doing a bit of EDA we choose a set of nominal, ordinal and numerical 
#columns to work with.

nominal = ["MSZoning", "LotShape", "LandContour", "LotConfig", "Neighborhood",
           "Condition1", "BldgType", "RoofStyle",
           "Foundation", "CentralAir", "SaleType", "SaleCondition"]

ordinal = ["LandSlope", "OverallQual", "OverallCond", "YearRemodAdd",
          "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
          "KitchenQual", "Functional", "GarageCond", "PavedDrive"]

numerical = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtUnfSF",
            "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea",
            "OpenPorchSF"]

train_features = train_df[nominal + ordinal + numerical]
train_label = train_df["SalePrice"]

val_features = val_df[nominal + ordinal + numerical]
val_label = val_df["SalePrice"]

test_features = test_df[nominal + ordinal + numerical]


#%%
#Preprocessing
#----------------------
#Now let's choose a preprocessing plan, a very straight forward one is the following:

# Ordinal features Impute missing data with most frequent valueUse Ordinal Encoding
# Impute missing data with most frequent value
# Use Ordinal Encoding

# Nominal Features Impute missing data with most frequent valueUse One Hot Encoding
# Impute missing data with most frequent value
# Use One Hot Encoding

# Numerical Features Impute missing data with mean valueUse Standard Scaling
# Impute missing data with mean value
# Use Standard Scaling

#As you may see, each family of features has its own unique way of getting 
#processed. Let's create a Pipeline for each family.

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

nominal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse=True, handle_unknown="ignore"))
])

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
# each of which has a the name of the preprocessor the transformation pipeline 
#(could be a transformer) and the list of column names we wish to transform

preprocessing_pipeline = ColumnTransformer([
    ("nominal_preprocessor", nominal_pipeline, nominal),
    ("ordinal_preprocessor", ordinal_pipeline, ordinal),
    ("numerical_preprocessor", numerical_pipeline, numerical)
])

## If you want to test this pipeline run the following code

# preprocessed_features = preprocessing_pipeline.fit_transform(train_features)

#%%
#Adding the model to the pipeline
#-----------------------------------
#Now that we're done creating the preprocessing pipeline let's add 
#the model to the end.

from sklearn.linear_model import LinearRegression

complete_pipeline = Pipeline([
    ("preprocessor", preprocessing_pipeline),
    ("estimator", LinearRegression())
])

#%%
#If you're waiting for the rest of the code, I'd like to tell you 
#that that's it. Pretty easy isn't it. If the scikit-learn maintainers 
#ask to take my heart I'd give it to them for such great API.

# The training and evaluation process is the same as any normal model

complete_pipeline.fit(train_features, train_label)
score = complete_pipeline.score(val_features, val_label)

print(score)

predictions = complete_pipeline.predict(test_features)

#%%
#Saving and Loading Pipelines
#-----------------------------------
#Now we want to save the entire preprocessing parameters and model 
#parameters of this pipeline to disk and load it whenever needed.

# We are going to use joblib for this JOB ... get it? ... sorry.

#Save the pipeline
#-----------------
#We are going to save the model as a pickle (.pkl) file. The code is fairly simple.

import joblib

pipeline_filename = "my_pipeline.pkl"
joblib.dump(complete_pipeline, pipeline_filename)

#%%
#Load the pipeline
#--------------------
#Now you're on your flask server and you wish to load the model to help 
#a user predict the price of a house, so you want to load the model from 
#disk when you start the server, or whenever a request is sent. That is
# also fairly simple.

import joblib

pipeline_filename = "path/to/pipeline/file.pkl"

pipeline = joblib.load(pipeline_filename)

## do inference with pipeline.predict