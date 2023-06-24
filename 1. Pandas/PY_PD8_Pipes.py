# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 04:26:54 2022

@author: rvamsikrishna
"""

#%%
#File-Contents
#--------------
#Pandas.DataFrame.pipe 
#The pipe() method allows you to apply one or more functions to the DataFrame object.

#https://www.nbshare.io/notebook/17251835/Summarising-Aggregating-and-Grouping-data-in-Python-Pandas/



#%%
#Custom fuctions to make work easier
import pandas as pd
import numpy as np

# Set seed
np.random.seed(520)

# Create a dataframe
df = pd.DataFrame({
    'name': ['Ted'] * 3 + ['Lisa'] * 3 + ['Sam'] * 3,
    'subject': ['math', 'physics', 'history'] * 3,
    'score': np.random.randint(60, 100, 9)
})

#%%
#To get rank by subject in a line

def get_subject_rank(input_df):
    # Avoid overwrite the original dataframe
    input_df = input_df.copy()
    input_df['subject_rank'] = (input_df
                                .groupby(['subject'])['score']
                                .rank(ascending=False))
    return input_df

# pipe method
df.pipe(get_subject_rank)

#%%
def add_score(input_df, added_score):
    # Avoid overwrite the original dataframe
    input_df = input_df.copy()
    input_df = input_df.assign(new_score=lambda x: x.score+added_score)
    return input_df

df.pipe(add_score, 2)

#%%
#Debug in method chaining
#************************
from functools import wraps
import logging

def log_shape(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logging.info("%s,%s" % (func.__name__, result.shape))
        return result
    return wrapper

def log_columns(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logging.info("%s,%s" % (func.__name__, result.columns))
        return result
    return wrapper

@log_columns
@log_shape
def get_subject_rank(input_df):
    input_df = input_df.copy()
    input_df['subject_rank'] = (input_df
                                .groupby(['subject'])['score']
                                .rank(ascending=False))
    return input_df

@log_columns
@log_shape
def add_score(input_df, added_score):
    input_df = input_df.copy()
    input_df = input_df.assign(new_score=lambda x: x.score+added_score)
    return input_df

(
    df.pipe(get_subject_rank)
      .pipe(add_score, 2)
)

#%%
INFO - get_subject_rank,(9, 4)
INFO - get_subject_rank,Index(['name', 'subject', 'score', 'subject_rank'], dtype='object')
INFO - add_score,(9, 5)
INFO - add_score,Index(['name', 'subject', 'score', 'subject_rank', 'new_score'], dtype='object')
#%%
#%%
#%%'
#Using Pandas pipe function to improve code readability
#https://towardsdatascience.com/using-pandas-pipe-function-to-improve-code-readability-96d66abfaf8

#Instead of writing
# f(), g(), and h() are user-defined function
# df is a Pandas DataFrame f(g(h(df), arg1=a), arg2=b, arg3=c)

#We can write
#(df.pipe(h)
#   .pipe(g, arg1=a)
#   .pipe(f, arg2=b, arg3=c)
#)

import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

def load_data():
    return pd.read_csv('data/train.csv')

df = load_data()
df.head()

#heat map missing values
sns.heatmap(df_train_raw.isnull(), 
            yticklabels=False, 
            cbar=False, 
            cmap='viridis')

#%%
#1. Split Name into first name and second name
def split_name(x_df):
    def split_name_series(string):
        firstName, secondName=string.split(', ')
        return pd.Series(
            (firstName, secondName),
            index='firstName secondName'.split()
        )    # Select the Name column and apply a function
    res=x_df['Name'].apply(split_name_series)
    x_df[res.columns]=res
    return x_df


res=(
    load_data()
    .pipe(split_name)
)

res.head()

#%%
#2. For Sex, substitute value male with M and female with F
def substitute_sex(x_df):
    mapping={'male':'M','female':'F'}
    x_df['Sex']=df['Sex'].map(mapping)
    return x_df

#x_df['Sex'] select the Sex column and then the Pandas map() used
#for substituting each value in a Series with another value.
    
res=(
    load_data()
    .pipe(split_name)
    .pipe(substitute_sex)
)

res.head()

#%%
#3. Replace the missing Age with some form of imputation

# we can be smarter about this and check the average age by passenger class. 

sns.boxplot(x='Pclass',
            y='Age',
            data=df,
            palette='winter')

pclass_age_map = { 1: 37,2: 29,3: 24}

def replace_age_na(x_df, fill_map):
    cond=x_df['Age'].isna()
    res=x_df.loc[cond,'Pclass'].map(fill_map)
    x_df.loc[cond,'Age']=res   
    return x_df

res=(
    load_data()
    .pipe(split_name)
    .pipe(substitute_sex)
    .pipe(replace_age_na, pclass_age_map)
)
res.head()

#CHECKING AGAIN FOR MISSING IN AGE
sns.heatmap(res.isnull(), 
            yticklabels=False, 
            cbar=False, 
            cmap='viridis')
#%%
#4. Convert ages to groups of age ranges: ≤12, Teen (≤18),
#Adult (≤60), and Older (>60)

#pd.cut() is used to convert ages to groups of age ranges.

def create_age_group(x_df):
    bins=[0, 13, 19, 61, sys.maxsize]
    labels=['<12', 'Teen', 'Adult', 'Older']
    ageGroup=pd.cut(x_df['Age'], bins=bins, labels=labels)
    x_df['ageGroup']=ageGroup
    return x_df

res=(
    load_data()
    .pipe(split_name)
    .pipe(substitute_sex)
    .pipe(replace_age_na, pclass_age_map)
    .pipe(create_age_group)
)

res.head()

#%%
#%%
#%%

#https://www.kdnuggets.com/2021/01/cleaner-data-analysis-pandas-pipes.html

def drop_missing(df):
    thresh = len(df) * 0.6
    df.dropna(axis=1, thresh=thresh, inplace=True)
    return df

def remove_outliers(df, column_name):
    low = np.quantile(df[column_name], 0.05)
    high = np.quantile(df[column_name], 0.95)
    return df[df[column_name].between(low, high, inclusive=True)]

def to_category(df):
    cols = df.select_dtypes(include='object').columns
    for col in cols:
        ratio = len(df[col].value_counts()) / len(df)
        if ratio < 0.05:
            df[col] = df[col].astype('category')
    return df

marketing_cleaned = (marketing.
                       pipe(drop_missing).
                       pipe(remove_outliers, 'Salary').
                       pipe(to_category))

#One important thing to mention is that the pipe function modifies
#the original dataframe. We should avoid changing the original 
#ataset if possible.

#To overcome this issue, we can use a copy of the original
#dataframe in the pipe. 

def copy_df(df):
   return df.copy()

marketing_cleaned = (marketing.
                       pipe(copy_df).
                       pipe(drop_missing).
                       pipe(remove_outliers, 'Salary').
                       pipe(to_category))

#%%
#%%
#%%






























































