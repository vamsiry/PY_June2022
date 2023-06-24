# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:39:26 2022

@author: rvamsikrishna
"""


#How to Use Sklearn Pipelines For Ridiculously Neat Code
#https://towardsdatascience.com/how-to-use-sklearn-pipelines-for-ridiculously-neat-code-a61ab66ca90d

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler

from sklearn.feature_selection import SelectKBest,chi2

from sklearn.pipeline import Pipeline,make_pipeline, FeatureUnion

#Difference between Pipeline and make_pipeline
#The pipeline requires the naming of steps while make_pipeline does not and 
#you can simply pass only the object of transformers. For the sake of 
#simplicity, you should use make_pipeline.


#FeatureUnion is another useful tool. It is capable of doing what
#ColumnTransformer just did but in a longer way:
#https://towardsdatascience.com/pipeline-columntransformer-and-featureunion-explained-f5491f815f

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV



train = pd.read_csv('data/train.csv')
X_test = pd.read_csv('data/test.csv')

train.iloc[:, 70:]

#%%

X = train.drop('SalePrice', axis=1)
y = train.SalePrice

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.3, 
                                                random_state=1121218)

#X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['total_bill', 'size']), 
#                                                    df['total_bill'], test_size=.2, random_state=seed)


#%%
X_train.describe().T.iloc[:10] # All numerical cols

X_train.describe().T.iloc[:10] # All numerical cols

#%%
above_0_missing = X_train.isnull().sum() > 0
X_train.isnull().sum()[above_0_missing]

#%%
numerical_features = X_train.select_dtypes(include='number').columns.tolist()
print(f'There are {len(numerical_features)} numerical features:', '\n')
print(numerical_features)

#%%
categorical_features = X_train.select_dtypes(exclude='number').columns.tolist()
print(f'There are {len(categorical_features)} categorical features:', '\n')
print(categorical_features)

#%%

numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


#Set handle_unknown to ignore to skip previously unseen labels. Otherwise, 
#OneHotEncoder throws an error if there are labels in test set that are 
#not in train set.


#sklearn.pipeline.Pipeline class takes a tuple of transformers for its
# steps argument. Each tuple should have this pattern:

#('name_of_transformer`, transformer)

#Then, each tuple is called a step containing a transformer like SimpleImputer
# and an arbitrary name. Each step will be chained and applied to the passed 
#DataFrame in the given order.

#%%
#But, these two pipelines are useless if we don’t tell which columns they 
#should be applied to. For that, we will use another transformer — ColumnTransformer.

numeric_pipeline.fit_transform(X_train.select_dtypes(include='number'))

categorical_pipeline.fit_transform(X_train.select_dtypes(include='object'))

#%%
#using the pipelines in this way means we have to call each pipeline separately
# on selected columns which is not what we want.

#To achieve this, we will use ColumnTransformer class:

from sklearn.compose import ColumnTransformer

full_processor = ColumnTransformer(transformers=[
    ('number', numeric_pipeline, numerical_features),
    ('category', categorical_pipeline, categorical_features)
])


#Similar to Pipeline class, ColumnTransformer takes a tuple of transformers. 
#Each tuple should contain an arbitrary step name, the transformer itself 
#and the list of column names that the transformer should be applied to.

#Now, we can use it to fully transform the X_train
full_processor.fit_transform(X_train)

#%%
#Final Pipeline With an Estimator
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

lasso = Lasso(alpha=0.1)

lasso_pipeline = Pipeline(steps=[
    ('preprocess', full_processor),
    ('model', lasso)
])


#Warning! The order of steps matter! The estimator should always be the 
#last step for the pipeline to work correctly.

#That’s it! We can now call lasso_pipeline just like we call any other model.
# When we call .fit, the pipeline applies all transformations before 
#fitting an estimator:

_ = lasso_pipeline.fit(X_train, y_train)

#%%
#Let’s evaluate our base model on the validation set (Remember, we have a 
#separate testing set which we haven’t touched so far):

preds = lasso_pipeline.predict(X_valid)

mean_absolute_error(y_valid, preds)

lasso_pipeline.score(X_valid, y_valid)


#%%
#Using Your Pipeline Everywhere

#we will use the pipeline in a grid search to find the optimal 
#hyperparameters in the next section.

#The main hyperparameter for Lasso is alpha which can range from 0 to infinity.
# For simplicity, we will only cross-validate on the values within 0 and 1 
#with steps of 0.05:

from sklearn.model_selection import GridSearchCV

param_dict = {'model__alpha': np.arange(0, 1, 0.05)}

search = GridSearchCV(lasso_pipeline, param_dict, 
                      cv=10, 
                      scoring='neg_mean_absolute_error')


_ = search.fit(X_train, y_train)

print('Best score:', abs(search.best_score_))

print('Best alpha:', search.best_params_)

#%%
#With the best hyperparameters, we get a significant drop in MAE (which is good). 
#Let’s redefine our pipeline with Lasso(alpha=76):

lasso = Lasso(alpha=76)

final_lasso_pipe = Pipeline(steps=[
    ('preprocess', full_processor),
    ('model', lasso)
])

#Fit it to X_train, validate on X_valid and submit predictions for the 
#competition using X_test:

_ = final_lasso_pipe.fit(X_train, y_train)

preds = final_lasso_pipe.predict(X_valid)

mean_absolute_error(y_valid, preds)

preds_final = final_lasso_pipe.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,'SalePrice': preds_final}
    
output.to_csv('submission.csv', index=False)



#%%
#%%
#%%
#%%
#*******************another type of transformer using pipeline**********************

#Create Column Transformer
#------------------------------
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[
    ('tnf1',SimpleImputer(),['fever']),
    ('tnf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
    ('tnf3',OneHotEncoder(sparse=False,drop='first'),['gender','city'])
],remainder='passthrough')

x_train_transform = transformer.fit_transform(X_train)


#%%
#Machine Learning Pipelines
#1st Imputation Transformer
trf1 = ColumnTransformer([
        ('impute_age',SimpleImputer(),[2]),
        ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])
    ],remainder='passthrough')

#2nd One Hot Encoding
trf2 = ColumnTransformer([
        ('ohe_sex_embarked', OneHotEncoder(sparse=False, handle_unknown='ignore'),[1,6])
    ], remainder='passthrough')

#3rd Scaling
trf3 = ColumnTransformer([
    ('scale', MinMaxScaler(), slice(0,10))
])

#4th Feature selection
trf4 = SelectKBest(score_func=chi2,k=8)

#5th Model
trf5 = DecisionTreeClassifier()



#4) Create Pipeline
pipe = Pipeline([
    ('trf1', trf1),
    ('trf2', trf2),
    ('trf3', trf3),
    ('trf4', trf4),
    ('trf5', trf5)
])

# Display Pipeline
from sklearn import set_config
set_config(display='diagram')
#fit data
pipe.fit(X_train, y_train)

#%%
#5) Cross-Validation using Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

params = {'trf5__max_depth':[1,2,3,4,5,None] }

grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(grid.best_score_)

#%%
#6) Exporting the Pipeline
# export 
import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))

#%%






















    
    