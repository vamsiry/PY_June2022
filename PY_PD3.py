# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 18:45:47 2022

@author: rvamsikrishna
"""
#%%
#File-Contents
#--------------
# All Column names 
# Numerical column names
# Categorical column names
# subsetting Numerical columns
# subsetting Categorical columns


#%%
#--selection)column selection)
#--slicing(row selection)
#--indexing(column & row selection)
#--filtering / Subsetting(selection based on conditions)
#--Sort Values


#https://towardsdatascience.com/23-efficient-ways-of-subsetting-a-pandas-dataframe-6264b8000a77

#Sorting and Subsetting in Python
#https://towardsdatascience.com/sorting-and-subsetting-in-python-f9dd2e14caa0

#%%
#%%
#%%
#%%
import pandas as pd

df = pd.read_csv("C://Users//rvamsikrishna//Desktop//PY//Python//PY Data Modeling//marketing_analysis.csv",skiprows=1, low_memory=False)

#url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/HairEyeColor.csv"
#df = pd.read_csv(url)

#%%

[*df] #List of column names
#%%
{*df} #dictionary of column names
#%%
*df, #Tuple of column names
#%%
*cols, = df  # A wild comma appears, again
cols
#%%
list(df)
#%%
sorted(df) #does not preserve the original order of the columns. 
#For that, you should use list(df) instead.
#%%
df.columns #returns pandas.core.indexes.base.Index not a list. use the .tolist()
#%%
df.keys() #returns index for Series, columns for DataFrame.
#%%
df.columns.tolist()
#%%
df.columns.values
#%%
df.columns.values.tolist()
#%%
df.columns.to_numpy().tolist()
#%%
df.keys().values.tolist()
#%%
[c for c in df] #gets all column names of df

#%%
df.head(1).columns
#or
list(df.head(1).columns)

#%%
df.columns[df.isnull().any()].tolist() #column names of missing values 

#%%
df.columns[df.isnull().any()].index #index of misssing values columns

#https://stackoverflow.com/questions/37366717/pandas-print-column-name-with-missing-values









#%%
#%%
#%%
#%%
#numeric columns names of data frame
#*************************************
numerics = ['int16', 'int32', 'int64', 'float64']
df.select_dtypes(include=numerics).columns
#%%
df.select_dtypes(include='number').columns
#%%
df.select_dtypes(include = np.number).columns
#%%
df.select_dtypes(include = [np.int64,np.float64]).columns
#%%
df._get_numeric_data().columns.values.tolist()
#%%
df.dtypes[df.dtypes == "int64"].index.values.tolist()
#%%
df.select_dtypes(exclude=['object']).columns.tolist()
#%%
df.describe().columns
#include = number,np.number,'int16','int32',np.int64,np.float64,"int64","object"








#%%
#%%
#%%
#%%
#%%
#Categorical column names of data frame
#******************************************
df.select_dtypes(include = np.object).columns
#%%
df.select_dtypes(include=['category','object']).columns
#%%
df.select_dtypes(include=['category','object']).dtypes
#%%
df.select_dtypes(include=['category','object']).columns.tolist()
#%%
df.select_dtypes(include=['object','object']).columns.tolist()
#%%
cat_features=[i for i in df.columns if df.dtypes[i]=='object']
cat_features
#%%
cat_cols = [col for col in df.columns if col not in df.describe().columns]
cat_cols
#%%
cat_col = [c for i, c in enumerate(df.columns) if df.dtypes[i] in [np.object]]
cat_col
#%%
df.dtypes[df.dtypes == 'object'].index
#%%
df.dtypes[df.dtypes.isin(['object','category'])].index
#%%
numeric_var = [key for key in dict(df.dtypes)
                   if dict(pd.dtypes)[key]
                       in ['float64','float32','int32','int64']] # Numeric Variable

cat_var = [key for key in dict(df.dtypes)
             if dict(df.dtypes)[key] in ['object'] ] # Categorical Varible







#%%
#%%
#%%
#%%
#%%
#selection(column selection)--slicing(row selection)--indexing(column & row selection)
#--filtering(selection based on conditions)
#***************************************************************************

#https://towardsdatascience.com/23-efficient-ways-of-subsetting-a-pandas-dataframe-6264b8000a77

# .loc() is based on the labels of rows and columns.
# .iloc() is based on an index of rows and columns

#%%
#%%
#%%
#Selection(column selection)
#***************************
#Method 1: Selecting a single column using the column name

df['age']
# Or
df.age # Only for single column selection

#%%
df.columns
#%%
#Method 2: Selecting multiple columns using the column names
df[['age', 'marital', 'jobedu']].head()

#%%
#The general syntax of the .loc attribute is:
#df.loc['row_label', 'column_label']

#If there are multiple labels, they should be specified inside lists:
#df.loc[['row_1', 'row_2'], ['column_1', 'column_2']]

#%%
#Method 3: Selecting a single column using the .loc attribute
df.loc[:, 'age']

#%%
#Method 4: Selecting multiple columns using the .loc attribute
df.loc[:, ['age', 'salay', 'marital']]

#%%
#The general syntax of the .iloc attribute is:
#df.iloc['row_index', 'column_index']

#If there are multiple labels, they should be specified inside lists:
#df.iloc[['row_index_1', 'row_index_2'], ['column_index_1', 'column_index_2']]

#%%
#Method 5: Selecting a single column using the .iloc attribute
df.iloc[:, 0]

#%%
#Method 6: Selecting multiple columns using the .iloc attribute
df.iloc[:, [0, 2, 10]]

#%%
#Method 7: Selecting consecutive columns using the .iloc attribute (The easy way)
df.iloc[:, [0, 1, 2, 3, 4]]

#or

df.iloc[:, 0:5]

#%%
#Method 8: Selecting the last column
df.iloc[:, -1]


#%%
#%%
#%%
#%%
#%%
#slicing(row selection)
#***********************
#Using iloc function
#-----------------------
#Method 9: Selecting a single row using the .iloc attribute
df.iloc[0]
#or
df.iloc[[0]] #for better view

#%%
#Method 10: Selecting multiple rows using the .iloc attribute
df.iloc[[0, 25, 100]]

#%%
#Method 11: Selecting the last few rows
df.iloc[[-1, -2, -3, -4, -5]]

#%%
#Using Loc function
#--------------------
df.loc[[1]] #1. Selecting 1st Rows with loc()
#%%
df.loc[[1,5,7]] #To select multiple rows (1,5,7) 
#%%
df.loc[1:7] #To select multiple rows (1:7) 
#%%
# Select all columns for rows of index values 0 and 10
df.loc[[0, 10], :]



#%%
df[:1] #1st row of a data set
#%%
df[:4] # first 1st 4 rows of the dataset
#%%
df[-1:] # Select the last row/element in the dataset/list
#%%
df[:-1] #excluding last row of the dataset
#%%
df[0:3] # Select rows 0, 1, 2 (row 3 is not selected)
#%%
df[:5] # Select the first 5 rows (rows 0, 1, 2, 3, 4)
#%%
# Assign the value `0` to the first three rows of data in the DataFrame
df[0:3] = 0






#%%
#%%
#%%
#indexing(column & row selection)
#************************************
#When we combine column selection and row slicing, it is referred to as Indexing. 
#Here, we can use .loc and .iloc attributes of a Pandas DataFrame.

#Method 12: Selecting a single value using the .iloc attribute
df.iloc[0, 0]

#%%
#Method 13: Selecting a single value using the .loc attribute
df.loc[0, 'age']

#%%
# Select all columns for rows of index values 0 and 10
df.loc[[0, 10], :]

#%%
#Method 14: Selecting multiple rows and columns using the .iloc attribute
df.iloc[[0, 5, 100], [0, 3, 7]]

#%%
#Method 15: Selecting multiple rows and columns using the .loc attribute
df.loc[[0, 5, 100], ['age', 'balance', 'housing']] #here keep in mind that row names are the same as the row indices.

#%%
#Method 16: Selecting consecutive rows and columns using the .loc and .iloc attributes (The easy way)
df.iloc[0:6, 0:5]

#%%
df.loc[0:6, ['age', 'housing']]
#%%








#%%
#%%
#%%
#filtering / Subsetting (selection based on conditions)
#****************************************
#Method 17: Filtering based on a single criterion with all columns
d1 = df[df['age'] > 14]
#%%
d2 = df[df.year == 2002]
#%%
d3 = df[df.year != 2002]
#%%
d4 = df[df['Zodiac Sign'] == 'Leo']
#%%
listGoesHere = ['a','b','c','d']
d4 = df[df['species_id'].isin([listGoesHere])]
#%%
df[df.Eye.isin(['Blue','Hazel','Green'])].head()
#%%
df[df.Eye.isin(['Blue','Hazel','Green'])].shape
#%%
df[df.Eye.isin(['Blue','Hazel']) & (df.Sex=="Male")].shape[0]

#%%
df.query("Zodiac Sign =='Leo'")


#%%
#Method 18: Filtering based on a single criterion with a few columns
d5 = df.loc[df['age'] > 14, ['age', 'marital', 'jobedu']]

#%%
#Method 19: Filtering based on two criteria with AND operator (Same column)
d6 = df[(df['age'] > 22) & (df['age'] < 29)]
#or
d7 = df[(df.year >= 1980) & (df.year <= 1985)]


#%%
#Method 20: Filtering based on two criteria with the between() method
df[df['age'].between(22, 29)]

#or
df[df['age'].between(22, 29, inclusive=False)]

#%%
#Method 21: Filtering based on two criteria with AND operator (Different columns)
df[(df['age'] > 25) & (df['balance'] > 300)]
#or
df.query("Zodiac Sign == 'Leo' & Sex =='Male'")

#%%
#Method 22: Filtering based on two criteria with OR operator
df[(df['age'] > 25) | (df['balance'] > 300)]

#%%
#Method 23: Filtering based on the minimum and maximum values
df['age'].idxmin() # Min value index number
#df['age'].idxmax() # Max value index number

#%%
df.loc[33699,'age'] # Max value index

#%%
df.iloc[[df['age'].idxmin(), df['age'].idxmax()]] #rows of min nd max age 

#%%
# select rows with NaN values, we can use the 'any()' method
df[pd.isnull(df).any(axis=1)]

#%%
df2 = df.loc[:, df.columns.isin(list('BCD'))]
#%%
df3  = df.loc[:, 'C':'E']
#%%
df1 = df.iloc[0,0:2].copy() # To avoid the case where changing df1 also changes df

#%%
# To select just the rows with NaN values in age column,
SUBSETTING_DF_WITH_NA_IN_AGE = df[pd.isnull(df['age'])]['age']
print(SUBSETTING_DF_WITH_NA_IN_AGE)

NA_INDEX_IN_AGE = df[pd.isnull(df['age'])]['age']
#%%







#%%
#%%
#%%
#%%
#%%
#Sorting Values
#----------------
df.sort_values(by = 'age', inplace = True)

#%%
df.sort_values(by='Name', inplace=True) #sorting df based on spefic column

#%%
df.sort_values('age',ascending = False)
#%%
df.sort_values(['marital','salary'])
#%%
df.sort_values(['marital','salaey'], ascending=[True, False])
#%%

df.plot(x='age',y='salary',kind = 'scatter')

#%%










#%%
#%%
#Index,Reindex,set_index,reset_index,

##pandas.DataFrame.reindex
#--------------------------------
index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']

df = pd.DataFrame({'http_status': [200, 200, 404, 404, 301],
                  'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},
                  index=index)
df
#%%
new_index = ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10','Chrome']
df.reindex(new_index)    
#%%
df.reindex(new_index, fill_value=0)
#%%
df.reindex(new_index, fill_value='missing')
#%%
df.reindex(columns=['http_status', 'user_agent'])
#%%
df.reindex(['http_status', 'user_agent'], axis="columns")
#%%
date_index = pd.date_range('1/1/2010', periods=6, freq='D')

df2 = pd.DataFrame({"prices": [100, 101, np.nan, 100, 89, 88]},
                   index=date_index)

df2
#%%
date_index2 = pd.date_range('12/29/2009', periods=10, freq='D')
df2.reindex(date_index2)

#%%
df2.reindex(date_index2, method='bfill')

#%%
#pandas.DataFrame.set_index
#---------------------------------
df = pd.DataFrame({'month': [1, 4, 7, 10],'year': [2012, 2014, 2013, 2014],
                   'sale': [55, 40, 84, 31]})

df
#%%
df.set_index('month')

#%%
df.set_index(['year', 'month'])

#%%
#Create a MultiIndex using an Index and a column:
df.set_index([pd.Index([1, 2, 3, 4]), 'year'])

#%%
#Create a MultiIndex using two Series:
s = pd.Series([1, 2, 3, 4])
df.set_index([s, s**2])

#%%
#pandas.DataFrame.reset_index¶
#----------------------------
#Reset the index of the DataFrame, and use the default one instead. 
#If the DataFrame has a MultiIndex, this method can remove one or more levels.

df = pd.DataFrame([('bird', 389.0),('bird', 24.0),('mammal', 80.5),('mammal', np.nan)],
                   index=['falcon', 'parrot', 'lion', 'monkey'],
                  columns=('class', 'max_speed'))

df
#%%
df.reset_index()
#%%
df.reset_index(drop=True)
#%%
#You can also use reset_index with MultiIndex.
index = pd.MultiIndex.from_tuples([('bird', 'falcon'),
                                   ('bird', 'parrot'),
                                   ('mammal', 'lion'),
                                   ('mammal', 'monkey')],
                                  names=['class', 'name'])

columns = pd.MultiIndex.from_tuples([('speed', 'max'),
                                     ('species', 'type')])

df = pd.DataFrame([(389.0, 'fly'),
                   ( 24.0, 'fly'),
                   ( 80.5, 'run'),
                   (np.nan, 'jump')],
                  index=index,  
                  columns=columns)

df
#%%
df.reset_index(level='class')
#%%
df.reset_index(level='class',drop=True)
#%%
#If we are not dropping the index, by default, it is placed in the top level.
# We can place it in another level:

df.reset_index(level='class', col_level=1)

#%%
#When the index is inserted under another level, we can specify under
# which one with the parameter col_fill:

df.reset_index(level='class', col_level=1, col_fill='species')

#%%
df.reset_index(level='class', col_level=1, col_fill='genus')

#%%


















