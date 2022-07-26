# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 08:59:36 2022

@author: rvamsikrishna
"""

#File-Contents
#--------------
# 1. Loading a file into data frame
# 2. All Basic EDA fro data understanding
# 3. Some basic hints at begining


#%%
#%%
#My top 25 pandas tricks
#----------------------------
#https://www.youtube.com/watch?time_continue=67&v=RlIiVeig3hc&feature=emb_logo
import pandas_profiling as pp
pp.ProfileReport(df) #to display the report

#%%
from platform import python_version
print(python_version())
#%%
pip --version

#%%
#loading libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

#%%
os.getcwd()


#%%
df = pd.read_csv("C://Users//rvamsikrishna//Desktop//PY//Python//PY Data Modeling//marketing_analysis.csv",skiprows=1, low_memory=False)

#url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/HairEyeColor.csv"
#df = pd.read_csv(url)

#%%
print(pd.options.display.max_rows)
#%%
pd.options.display.max_rows   = 15

#%%
df.memory_usage()
#%%
type(df)
#%%
df.shape # num of Rows & Columns
#%%
df.ndim
#%%
s.ndim #s-series
#%%
df.age.size #no.of elements in df or a col
#%%
df.head(10)
#%%
df.tail()
#%%
df.sample(10) 
#%%
print(df.to_string()) #To print the entire DataFrame.
#%%
df.dtypes #print data type of all columns
#%%
#Convert columns to best possible dtypes using dtypes supporting pd.NA.
df2 = df.convert_dtypes()
#%%
df["age"] = df['Customer Number'].astype('int')
#%%
df["Customer Number"] = df['Customer Number'].astype('str')
#%%
df['Is_Male'] = df.Is_Male.astype('category')
#%%
df["IsPurchased"] = df['IsPurchased'].astype('bool')
#%%
df["Total Spend"] = df['Total Spend'].astype('float')
#%%
df['Dates'] = pd.to_datetime(df['Dates'], format='%Y%m%d')
#%%
# alternatively, pass { col: dtype }
df = df.astype({'price': 'int'})
#%%
# use to_numeric to convert the strings with Invalid characters
df['sales'] = pd.to_numeric(df['sales'], errors='coerce')


#%%
df.infer_objects().dtypes #Attempt to infer better dtypes for object columns.
#%%
#to_datetime --  Convert argument to datetime.
#to_timedelta -- Convert argument to timedelta.
#to_numeric -- Convert argument to numeric type.


#%%
df.info() # prints data type and missing values for each column

#%%

#To select all numeric types, use np.number or 'number'
#To select strings you must use the object dtype, but note that this will return all object dtype columns
#To select Pandas categorical dtypes, use 'category'
#To select datetimes, use np.datetime64, 'datetime' or 'datetime64'
#To select timedeltas, use np.timedelta64, 'timedelta' or 'timedelta64'
#To select Pandas datetimetz dtypes, use 'datetimetz' (new in 0.20.0) or 'datetime64[ns, tz]'
#See the numpy dtype hierarchy


#include = number,np.number,,np.int64,np.float64,
#'category','object','float64','float32','int16','int32','int64'

df.describe() # Summary Statistics

#%%
df.age.describe()

#df.describe(include=[np.number])
#df.describe(exclude=[np.number])  

#Numeric = ['number',np.number,,np.int64,np.float64,'float64','float32','int16','int32','int64']
#Categorical = ['category','object']

#df.describe(include = Numeric ) # Summary Statistics
#df.describe(exclude = Categorical ) # Summary Statistics

#df.describe(percentiles = [.75,.85,.95] ) # Summary Statistics
#df.age.describe(percentiles = [.20,.40,.60,.80,.90] ) # Summary Statistics

#df.describe().T.iloc[:10] # All numerical cols
#df.describe(include=np.object).T.iloc[:10] # All object cols


#%matplotlib inline
#df.describe().plot()

df.describe(percentiles=[0, 1/3, 2/3, 1])

#%%
df.describe(include=[np.number]).plot()

#%%
df.describe(include=[['category','object']]).plot()

#%%
# Unique Eye colors
df["marital"].unique()

#%%
df['marital'].nunique()

#%%
df.nunique(axis=0) #unique values counts in each column

#%%
df.response.value_counts() # target class distribution

#%%
#Proportion of each castegory in the column jobedu
df.jobedu.value_counts(normalize=True) 

#%%
#plot the bar graph of percentage job categories
df.job.value_counts(normalize=True).plot.barh()
plt.show()

#%%
#calculate the percentage of each education category.
df.education.value_counts(normalize=True)

#%%
#plot the pie chart of education categories
df.education.value_counts(normalize=True).plot.pie()
plt.show()

#%%
#Drop a column
df.drop('SalePrice', axis=1)
#%%
df.drop(['SalePrice','age'], axis=1)

#%%
#Get item from object for given key (ex: DataFrame column).
df.get(["temp_celsius", "windspeed"])
   
#%%
#Return a booloian index of true and false
df.duplicated()
 
#%%
#removes duplicate rows
df.drop_duplicates(inplace=True) 

#%%
# Unique rows based on 2 columns eye and sex
df[["Eye","Sex"]].drop_duplicates()

#%%
len(df.index) - len(df.drop_duplicates())

#%%
#Mapping categories to required values
df['GenderMap'] = df.Gender.map({'Male':1,'Female':0},na_action='ignore')

#%%
titles_dict = {'Capt': 'Other','Major': 'Other', 'Jonkheer': 'Other',
               'Don': 'Other','Sir': 'Other','Dr': 'Other','Rev': 'Other',
               'Countess': 'Other','Dona': 'Other','Mme': 'Mrs','Mlle': 'Miss',
               'Ms': 'Miss','Mr': 'Mr','Mrs': 'Mrs','Miss': 'Miss',
               'Master': 'Master','Lady': 'Other'}

df['Title'] = df['Title'].map(titles_dict)
df['Title'] = pd.Categorical(df['Title'])

#%%
conversion_dict = {1: 'bin1',
                   2: 'bin2',
                   3: 'bin3',
                   4: 'bin4',
                   5: 'bin5',
                   6: 'bin6',
                   7: 'bin7'}

bins = list(map(conversion_dict.get, df['age']))


#%%
s = pd.Series(['cat', 'dog', np.nan, 'rabbit'])
s    
s.map({'cat': 'kitten', 'dog': 'puppy'},na_action='ignore')

#%%
#map accepts a dict or a Series. Values that are not found in the dict
#are converted to NaN, unless the dict has a default value (e.g. defaultdict):
    
#It also accepts a function:
s.map('I am a {}'.format)
s.map('I am a {}'.format, na_action='ignore')

#%%
# Rename columns
df.rename(columns = {"Freq":"Frequency","Eye":"Eye_Color"},inplace=True)

#%%
#numeric column names 
df.select_dtypes(include = [np.int64,np.float64]).columns.values.tolist()

#%%
#categorical column names
df.select_dtypes(include = [np.object]).columns.values.tolist()

#%%
#Subsetting Numerical data
numerical_data = df.select_dtypes(include = [np.int64,np.float64])

#%%
#Subsetting categorical data
numerical_data = df.select_dtypes(include = [np.object])

#%%
#6. count(): Return number of non-NA/null observations.
#Can be applied to both dataframe and series:
df.count()

df.count(numeric_only = True)

df.marital.count()

#%%
#%%
#df.where() #Replace values where the condition is False.
#df.mask() #Replace values where the condition is True.

s = pd.Series(range(10))
print(s)
#%%
s.where(s < 5)
#%%
s.mask(s < 5)
#%%
s.where(s > 1, 10)
#%%
s.mask(s > 1, 10)
#%%


#%%
#%%
#pandas.cut¶
#***************
#Create bins and bin up postTestScore by those pins
df = pd.DataFrame({'age': [2, 67, 40, 32, 4, 15, 82, 99, 26, 30, 50, 78]})
df
#%%
df['age_group'] = pd.cut(df['age'], 3)
df
#%%
df['age_group'].value_counts().sort_values()
#%%
df['age_group'] = pd.cut(df['age'], 3, labels=["bad", "medium", "good"])
df
#%%
df['age_group'] = pd.cut(df['age'], bins=4, labels=False)
df
#%%
df['bins'] = pd.cut(x=df['age'], bins=[1, 20, 40, 60,80, 100])
print(df)
#print(df['bins'].unique())
#%%
df['bins2'] = pd.cut(x=df['age'], bins=[1, 20, 40, 60, 80, 100],
                    labels=['1 to 20', '21 to 40', '41 to 60',
                            '61 to 80', '81 to 100'])    
df

#%%
#%%



#%%
#%%%
#Pandas qcut()
#**************
df = pd.DataFrame({'age': [2, 67, 40, 32, 4, 15, 82, 99, 26, 30, 50, 78]})
df
#%%
df['age_group'] = pd.qcut(df['age'], 3)
df['age_group'].value_counts().sort_values()

#%%
#2. Discretizing into buckets with a list of quantiles
df['age_group'] = pd.qcut(df['age'], [0, .1, .5, 1])
df.sort_values('age_group')['age_group'].value_counts()

#%%
# Same as pd.qcut(df['age'], 4)
df['age_group'] = pd.qcut(df['age'], [0, .25, .5, .75, 1])
df['age_group'].value_counts().sort_values()

#%%
#3. Adding custom labels
labels=['Millennial', 'Gen X', 'Boomer', 'Greatest']
df['age_group'] = pd.qcut(df['age'], [0, .1, 0.3, .6, 1], labels=labels)

df['age_group'].value_counts().sort_index()

#%%
#4. Returning bins with retbins=True
# It is useful when q is passed as a single number value
result, bins = pd.qcut(df['age'], 5, retbins=True)
bins

#%%
#5. Configuring the bin precision with precision
# We can set the precision to 0 to avoid any decimal place.
pd.qcut(df['age'], 3, precision=0)

#%%
#%%
#6. Build a DataFrame from multiple files
from glob import glob

##For row-wise
#----------------
files = sorted(glob('data/data_row_*.csv'))
pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

#%%
#For column-wise
#------------------
files = sorted(glob('data/data_col_*.csv'))
pd.concat((pd.read_csv(file) for file in files), axis=1)






































































































#%%
df.index

#%%
#Columns names of numeric,category,missing data col, etc
#---------------------------------------------------------
df.columns
#%%
len(df.columns)
#%%
df.columns.to_list()
#%%
df.columns[df.isnull().any()].tolist() #column names of missing values 
#%%
df.columns[df.isnull().any()].index #index of misssing values columns
#%%
#numeric columns names of data frame

#include = number,np.number,,np.int64,np.float64,
#'category','object','float64','float32','int16','int32','int64'

numerics = ['int16', 'int32', 'int64', 'float64']

df.select_dtypes(include=numerics).columns.tolist()

df.select_dtypes(include=['category','object']).columns.tolist()

df.select_dtypes(exclude=['object']).columns.tolist()

#%%




#%%
# use 'loc' for index name and 'iloc' for index position number
df.loc[1] #Prints 1st row and all columns of a data frame
#%%
df.loc[[1,3,6,9]] #Prints list of rows and all columns of a data frame
#%%
df.loc[1:10] #Prints 1st row and all columns of a data frame
#%%
df.loc[len(df)-10:len(df)-1] #Prints 1st row and all columns of a data frame
#%%
df.loc[:, 'age']
#%%
df.loc[[1,4,7,44,99], ['customerid', 'Age', 'salary']]
#%%
df.loc[0:4, ['customerid', 'Age', 'salary']]
#%%


#%%
df.iloc[1] #1st row
#%%
df.iloc[[1,2,3]] #list of rows
#%%
df.iloc[[-1, -2, -3, -4, -5]] #selecting last few rows
#%%
df.iloc[1:10] #Prints 1st row and all columns of a data frame
#%%
df.iloc[len(df)-10:len(df)-1] #Prints 1st row and all columns of a data frame
#%%
df.iloc[:, 0]
#%%
df.iloc[:, [0, 2, 10]]
#%%
df.iloc[:, 0:5]
#%%
df.iloc[[1,4,7,44,99], [1,4,6,8,99]]
#%%
df.iloc[0:4,0:4]
#%%
df.iloc[1,1] 
#%%
X = df.iloc[:, :-1].values #all the rows and except last column of the data set

#That first colon (:) means that we want to take all the rows in our dataset.
# : -1 means that we want to take all of the columns of data except the last one.
# The .values on the end means that we want all of the values.

y = df.iloc[:, -1].values # all the row but only last column of the dataset(tarfet column)

#%%






