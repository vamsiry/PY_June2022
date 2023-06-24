# -*- coding: utf-8 -*-
"""
Created on Tue May  3 06:53:04 2022

@author: rvamsikrishna
"""

#isnull()
#notnull()
#dropna()
#fillna()
#replace()
#interpolate()

#isnull() and isna() literally does the same things. isnull() 
#is just an alias of the isna() method as shown in pandas source code

# Possible solution fro missing values are as below
# 1.DROP MISSING VALUES
# 2.FILL MISSING VALUES WITH TEST STATISTIC
# 3.PREDICT MISSING VALUE WITH A MACHINE LEARNING ALGORITHM

    
import pandas as pd 
import numpy as np

#%%
#df = pd.read_csv('C:\\Users\\rvamsikrishna\\Desktop\\PY\\Python\\Models\\Classification\\C Churn\\CCHURN.csv')
df = pd.read_csv('C:\\Users\\rvamsikrishna\\Desktop\\PY\\Python\\PY Data Modeling\\Missing_data.csv')


#%%
df.info() # each data type of columns and missing values


#%%
#Total missing values in the data set 
#****************************************
#np.count_nonzero(df.isnull().values) #Total missing:
#or
#df.isnull().values.ravel().sum() #Total missing:
#or
df.isnull().sum().sum() #Total missing:
#or
#df.isnull().values.sum()


#%%
# No.of columns having missing values
#****************************************
df.isnull().any(axis=0).sum()
#or 
#(df.isna().sum(axis=0) > 0).sum()

#%%

#%%
#Count NaN values under a single DataFrame column:
#df['Gender'].isna().sum()
#or
df['Gender'].isnull().sum()

#%%
# No.of rows having least one missing values
#****************************************
df.isnull().any(axis=1).sum()
#or
#(df.isna().sum(axis=1) > 0).sum()
#or
#df.shape[0] - df.dropna().shape[0]
#or
#sum(df.apply(lambda x: sum(x.isnull().values), axis = 1)>0)


#%%
# No.of missing values in each Column 
#****************************************
df.isnull().sum()

#df.isnull().sum(axis = 0).sort_values(ascending = False).head(5)
#or
#print(" \nCount total NaN at each column in a DataFrame : \n\n",df.isnull().sum())
#or
#df.apply(lambda x: sum(x.isnull().values), axis = 0) #missing values in each columns

#%%
## Count total zeros at each column in a DataFrame
df[df == 0].count(axis=0)


#%%
# Percent of missing values in each column
#*******************************************
percent_missing = df.isnull().sum() * 100 / len(df)

missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'missing values count' :df.isnull().sum(),
                                 'percent_missing': percent_missing})
missing_value_df

#or 
#df.isnull().mean().round(4).mul(100).sort_values(ascending=False)

#%%
#or
df.isna().mean()

#or
#%%
# Percent of missing values in each column besed on target column
#*******************************************************************
gdf = df.groupby(['Gender'])

def countna(x):
    return (x.isna()).sum()

def countnap(x):
    return (x.isna().sum()/len(x) * 100)

#gdf.agg(['count', countna, 'size']).T

df.groupby('Gender').agg([countna,countnap,'count','size']).T
    


#%%
#list of column names having missing values 
#****************************************
df.columns[df.isnull().any()].tolist() 

#%%
# list of indexes of columns having misssing values
#***********************************************
xx = df.columns[df.isnull().any()].tolist()

df.columns.get_indexer(xx)
    
#%%
#%%
#%%
#dropping all the rows with at least one Nan value (Null value)
df.dropna()
#or
df.dropna(axis = 0, how ='any') 


#%%
#Dropping the rows if all values in that row are missing.
df.dropna(how = 'all')

#%%
#Dropping columns with at least 1 null value.
df.dropna(axis = 1)

#%%
#dropping the columns that have >= 60% of missing values.
thresh = len(df) * 0.6
df.dropna(axis=1, thresh=thresh, inplace=True)

#or
for col, val in df.iteritems():
    if (df[col].isnull().sum() / len(val) * 100) > 30:
        df.drop(columns=col, inplace=True)

     
#%%
#%%
#%%
#If the missing value isn’t identified as NaN (showing as zero), then we
#have to first convert or replace such non NaN entry with a NaN.
df[‘column_name’].replace(0, np.nan, inplace= True)    

  
#%%
#%%
#%%
#df.fillna(value, method, axis, inplace, limit, downcast)

# Filling all missing value using with zero
df.fillna(0)

#%%
# filling all the  missing value with previous ones  
df.fillna(method ='pad')
#or
df.Team.fillna(method ='pad')

#%%
#Last observation carried forward (LOCF)
df["Age"] = df["Age"].fillna(method ='ffill',axis = 0)

##one can also specify an axis to propagate (1 is for rows and 0 is for columns)

#%%
# Filling a missing value with next ones  
df["Gender"] = df["Gender"].fillna(method ='bfill')

#%%
# Fill all the null values in Gender column with “No Gender”
df["Gender"] = df["Gender"].fillna("No Gender", inplace = True) 

#%%
# To replace NaN values with the mean
df['Age'].fillna(value=df['Age'].mean(), inplace=True)

#%%
# Filling a null values using replace() method
# will replace  Nan value in dataframe with value -99  
df.replace(to_replace = np.nan, value = -99) 


#%%
#Using interpolate() function to fill the missing values using linear method.
df.interpolate(method ='linear', limit_direction ='forward')

#Let’s interpolate the missing values using Linear method. Note that Linear 
#method ignore the index and treat the values as equally spaced.



#%%
#mean imputation---
#*********************
df.fillna(df.mean(), inplace=True)
#or
df["age"] = df.age.fillna(df['age'].mean())

#df["Age"] = df["Age"].replace(np.NaN, dataset["Age"].mean())

#df["Age"] = df["Age"].replace(np.NaN, dataset["Age"].mean())

#%%
#Median Imputation--
#*********************
#df.fillna(df.median(), inplace=True)
#or
df["age"] = df["age"].fillna(df["age"].median())

#df["Age"] = df["Age"].replace(np.NaN, dataset["Age"].median())


#%%
#mode imputation-- for categorical column (index[0] is high freq value in that column)
#*********************
df['Gender'] = df['Gender'].fillna(df['Gender'].value_counts().index[0])


# filling missing values with mode of column values
df.fillna(df.mode(), inplace=True)
df.sample(10)

#or 
#Mode - missed value
import statistics

df["Gender"] = df["Gender"].fillna(statistics.mode(df["Gender"]))

df["Gender"] = df["Gender"].replace(np.NaN, statistics.mode(df["Gender"]))


#%%
# import modules
#strategy=most_frequent / mean / median

from numpy import isnan
from sklearn.impute import SimpleImputer

value = df.values
 
# defining the imputer
imputer = SimpleImputer(missing_values=nan,
                        strategy='most_frequent') #mean,median
 
# transform the dataset
transformed_values = imputer.fit_transform(value)
 
# count the number of NaN values in each column
print("Missing:", isnan(transformed_values).sum())


#%%
# pandas fill na with value from another column 
df['Cat1'].fillna(df['Cat2'])


#%%
#Imputation using for loop
cateogry_columns=df.select_dtypes(include=['object']).columns.tolist()
integer_columns=df.select_dtypes(include=['int64','float64']).columns.tolist()

for column in df:
    if df[column].isnull().any():
        if(column in cateogry_columns):
            df[column]=df[column].fillna(df[column].mode()[0])
        else:
            df[column]=df[column].fillna(df[column].mean)



#%%
#%%
#%%
#********************** KNN Imputation**************************
#********************************************************************

from sklearn.impute import KNNImputer

#The default distance measure is a Euclidean distance measure that is NaN aware, 
#e.g. will not include NaN values when calculating the distance between members 
#of the training dataset. This is set via the “metric” argument.

#The number of neighbors is set to five by default and can be configured by the “n_neighbors” argument.
#Finally, the distance measure can be weighed proportional to the distance between
# instances (rows), although this is set to a uniform weighting by default, 
#controlled via the “weights” argument.
#KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')

#One thing to note here is that the KNN Imputer does not recognize text data values.
# It will generate errors if we do not change these values to numerical values. 

#A good way to modify the text data is to perform one-hot encoding or 
#create “dummy variables”. 

#The idea is to convert each category into a binary data column by assigning a 1 or 0. 

#Other options would be to use LabelEncoder or OrdinalEncoder
# from Scikit-Learn’s preprocessing package.

#First, we will make a list of categorical variables with text data and generate
# dummy variables by using ‘.get_dummies’ attribute of Pandas data frame package. 
#An important caveat here is we are setting “drop_first” parameters as True 
#in order to prevent the Dummy Variable Trap.

cat_variables = df[[‘Sex’, ‘Embarked’]]
cat_dummies = pd.get_dummies(cat_variables, drop_first=True)
cat_dummies.head()

#Next, we will drop the original “Sex” and “Embarked” columns from the data frame 
#and add the dummy variables.

df = df.drop(['Sex', 'Embarked'], axis=1)
df = pd.concat([df, cat_dummies], axis=1)
df.head()

#Another critical point here is that the KNN Imptuer is a distance-based imputation
# method and it requires us to normalize our data. Otherwise, the different scales 
#of our data will lead the KNN Imputer to generate biased replacements for the
# missing values. For simplicity, we will use Scikit-Learn’s MinMaxScaler which 
#will scale our variables to have values between 0 and 1.

from sklearn.preprocessing import MinMaxScalerscaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include='number')), 
                  columns = df.select_dtypes(include='number').columns)
df.head()

#Imputation
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

df.isna().sum()#checking

#%%
#%%
#%%
#*************************Imputation using Regression******************
#***********************************************************************
#the null values in one column are filled by fitting a regression model 
#using other columns in the dataset.
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

train_df_non_missing = df[df['Age'].isnull()==False]
test_df_missing = df[df['Age'].isnull()==True]

y = train_df_non_missing['Age']
train_df_non_missing.drop("Age",axis=1,inplace=True)

lr.fit(train_df_non_missing,y)


test_df_missing.drop("Age",axis=1,inplace=True)

pred = lr.predict(test_df_missing)
test_df_missing['Age']= pred

train_df_non_missing['Age'] = y

newdf = pd.concat([traindf, testdf], axis=1)

#sometimes, using models for imputation can result in overfitting the data.
#There is no perfect way for filling the missing values in a dataset.
# You have to experiment through different methods, to check which 
#method works the best for your dataset.

#%%
#%%
#%%
#*************************Imputation using Groupby******************
#***********************************************************************

df['value'] = df['value'].fillna(df.groupby('name')['value'].transform('mean'))


#or
df["value"] = df.groupby("name").transform(lambda x: x.fillna(x.mean()))
#or
df['value'] = df.groupby(['category', 'name'])['value'].transform(lambda x: x.fillna(x.mean()))
#or
df[['value', 'other_value']] = df.groupby(['category', 'name'])['value', 'other_value']\
    .transform(lambda x: x.fillna(x.mean()))



#or    
df['value1']=df.groupby('name')['value'].apply(lambda x:x.fillna(x.mean()))
#or
df['value']=df.groupby(['name','class'])['value'].apply(lambda x:x.fillna(x.mean()))
#or
df['value','other_value'] = df.groupby(['team','class'])['value','other_value']\
    .apply(lambda x: x.fillna(x.mean()))

#lambda x: x.fillna(x.mode().iloc[0])
    
#%%    










#%%
#%%
#%%
#WRAP UP WITH ASSERT
#*********************

#one can use assert to programmatically check that no missing or unexpected
# ‘0’ value is present. This gives confidence that code is running properly.

#Assert will return nothing is the assert statement is true and will return 
#an AssertionError if statement is false

#fill null values with 0
df=train.fillna(value=0)

##assert that there are no missing values
assert pd.notnull(df).all().all()
#%%
#or for a particular column in df
assert df.column_name.notall().all()
#%%
#assert all values are greater than 0
assert (df >=0).all().all()
#%%
#assert no entry in a column is equal to 0
assert (df['column_name']!=0).all().all()


#%%
#%%
#pd.get_value()/pd.set_value()
#------------------------------
def test_get_value(df):
    for i in df.index:
        val = df.get_value(i,'Age')
        if math.isnan(val):
            df.set_value( i,'Age',-1)

test_get_value(titanic_df)

#%%
df1 = pd.DataFrame({'a':[1,2,np.nan], 'b':[np.nan,1,np.nan]}) 
def info_as_df (df):
    null_counts = df.isna().sum()
    info_df = pd.DataFrame(list(zip(null_counts.index,null_counts.values))\
                                         , columns = ['Column', 'Nulls_Count'])
    data_types = df.dtypes
    info_df['Dtype'] = data_types.values
    return info_df
print(df1.info())
print(info_as_df(df1))

#%%