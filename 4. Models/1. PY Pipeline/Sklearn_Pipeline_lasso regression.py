# -*- coding: utf-8 -*-
"""
Created on Wed May 18 21:17:31 2022

@author: rvamsikrishna
"""

#https://towardsdatascience.com/how-to-use-sklearn-pipelines-for-ridiculously-neat-code-a61ab66ca90d


train = pd.read_csv('data/train.csv')
X_test = pd.read_csv('data/test.csv')

train.iloc[:, 70:]

from sklearn.model_selection import train_test_split

X = train.drop('SalePrice', axis=1)
y = train.SalePrice

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.3, 
                                                random_state=1121218)

#%%
X_train.describe().T.iloc[:10] # All numerical cols

X_train.describe(include=np.object).T.iloc[:10] # All object cols

#%%
above_0_missing = X_train.isnull().sum() > 0

X_train.isnull().sum()[above_0_missing]

#%%
numerical_features = X_train.select_dtypes(include='number').columns.tolist()
print(f'There are {len(numerical_features)} numerical features:', '\n')
print(numerical_features)

categorical_features = X_train.select_dtypes(exclude='number').columns.tolist()
print(f'There are {len(categorical_features)} categorical features:', '\n')
print(categorical_features)

#%%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline


numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


#Set handle_unknown to ignore to skip previously unseen labels. Otherwise,
# OneHotEncoder throws an error if there are labels in test set that are not 
#in train set.

#sklearn.pipeline.Pipeline class takes a tuple of transformers for its steps 
#argument. Each tuple should have this pattern:
#('name_of_transformer`, transformer)

#%%
#Column Transformer
#-------------------
from sklearn.compose import ColumnTransformer

full_processor = ColumnTransformer(transformers=[
    ('number', numeric_pipeline, numerical_features),
    ('category', categorical_pipeline, categorical_features)
])


full_processor.fit_transform(X_train)

#Note that most transformers return numpy arrays which means index and column 
#names will be dropped.

#%%
#Final Pipeline With an Estimator
#--------------------------------------
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

lasso = Lasso(alpha=0.1)

lasso_pipeline = Pipeline(steps=[
    ('preprocess', full_processor),
    ('model', lasso)
])


#Warning! The order of steps matter! The estimator should always be the last step for the pipeline to work correctly.

#the pipeline applies all transformations before fitting an estimator:

_ = lasso_pipeline.fit(X_train, y_train)


preds = lasso_pipeline.predict(X_valid)
mean_absolute_error(y_valid, preds)

lasso_pipeline.score(X_valid, y_valid)

#%%
#Using Your Pipeline Everywhere
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
#Let’s redefine our pipeline with Lasso(alpha=.95):

lasso = Lasso(alpha=.95)

final_lasso_pipe = Pipeline(steps=[
    ('preprocess', full_processor),
    ('model', lasso)
])

#%%
_ = final_lasso_pipe.fit(X_train, y_train)

preds = final_lasso_pipe.predict(X_valid)

mean_absolute_error(y_valid, preds)

preds_final = final_lasso_pipe.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_final})

output.to_csv('submission.csv', index=False)

#%%






















