# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:04:07 2022

@author: rvamsikrishna
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import math
%matplotlib inline

#%%
df = pd.read_csv('C:/Users/rvamsikrishna/Desktop/PY/Python/C Churn/CCHURN.csv')

#%%
# create a dictionary with five fields each 
data = {
    'A':['A1', 'A2', 'A3', 'A4', 'A5'], 
    'B':['B1', 'B2', 'B3', 'B4', 'B5'], 
    'C':['C1', 'C2', 'C3', 'C4', 'C5'], 
    'D':['D1', 'D2', 'D3', 'D4', 'D5'], 
    'E':['E1', 'E2', 'E3', 'E4', 'E5'] }
  
# Convert the dictionary into DataFrame 
df = pd.DataFrame(data)
  
#%%
#**********************Dropping Columns from dataframe***********************
#****************************************************************************
#Method #1: Drop Columns from a Dataframe using drop() method.

#Dropping a column from pandas dataframe
df.drop("A", axis=1, inplace=True)

#%%
#Dropping multiple columns columns from dataframe using col name
df.drop(['C', 'D'], axis = 1, inplace = True)

#%%
# #Dropping multiple columns columns from dataframe using col index 
df.drop(df.columns[[0, 4, 2]], axis = 1, inplace = True)

#%%
df.drop(columns=df.columns[22:]) #Remove all columns after the 22th. 

#%%
#Method #2: Drop Columns from a Dataframe using iloc[] and drop() method.

# Remove all columns between column index 1 to 3
df.drop(df.iloc[:, 1:3], axis = 1, inplace = True)

#%%
#Method #3: Drop Columns from a Dataframe using ix() and drop() method.

# Remove all columns between column name 'B' to 'D'
df.drop(df.ix[:, 'B':'D'].columns, axis = 1)

#.ix is deprecated. Please use .loc for label based indexing or .iloc for positional indexing

#%%
#Method #4: Drop Columns from a Dataframe using loc[] and drop() method.

# Remove all columns between column name 'B' to 'D'
df.drop(df.loc[:, 'B':'D'].columns, axis = 1)

#%%
#Method #5: Drop Columns from a Dataframe by iterative way.
for col in df.columns:
    if 'A' in col:
        del df[col]
  
df

#%%
for col in df.columns:
    if 'Unnamed' in col:
        #del df[col]
        print col
        try:
            df.drop(col, axis=1, inplace=True)
        except Exception:
            pass


#%%
df.drop([col for col in df.columns if "Unnamed" in col], axis=1, inplace=True)

#%%
df = df[[col for col in df.columns if not ('Unnamed' in col)]]

#%%
to_remove = ['A','C']
df = df[df.columns.difference(to_remove)]
df

#%%
#%%
#%%
#%%
#%%
#*********************Apply function on a dataframe****************************
#******************************************************************************

#Dataframe/series.apply() is a function to single or selected columns or rows in Dataframe. 

#Method 1: Using Dataframe.apply() and lambda function.
import pandas as pd
import numpy as np
  
# List of Tuples
matrix = [(1, 2, 3),
          (4, 5, 6),
          (7, 8, 9)
         ]
  
# Create a DataFrame object
df = pd.DataFrame(matrix, columns = list('xyz'), index = list('abc'))  
print(df)

#%%
# Apply numpy.square() to lambda to find the squares of the values 
#for columns
new_df = df.apply(lambda x: np.square(x) if x.name == 'z' else x)
new_df


#%%
#Example 2: For Row.
new_df = df.apply(lambda x: np.square(x) if x.name == 'b' else x, axis = 1)
new_df  


#%%
#Method 2: Using Dataframe/series.apply() & [ ] Operator.
df['z'] = df['z'].apply(np.square)
df

#%%
#Example 2: For Row.
df.loc['b'] = df.loc['b'].apply(np.square)
df

#%%
#Method 3: Using numpy.square() method and [ ] operator.
df['z'] = np.square(df['z'])
df

#%%
#Example 2: For Row.
df.loc['b'] = np.square(df.loc['b'])
df

#%%
#%%
#We can also apply a function to more than one column or row in the dataframe.
#Example 1: For Column

new_df = df.apply(lambda x: np.square(x) if x.name in ['x', 'y'] else x)
new_df

#%%
#Example 2: For Row.
new_df = df.apply(lambda x: np.square(x) if x.name in ['b', 'c'] else x,axis = 1)
new_df

#%%
#Example 4: For Column
df[['x','z']] = df[['x','z']].apply(np.square)
df

#%%
#Example 6: For Column-Using numpy.square() method and [[]] operator.

df[['x','z']] = np.square(df[['x','z']])
df

#%%
#Example 5: For rows
df.loc[['a','b']] = df.loc[['a','b']].apply(np.square)
df


#%%
#Example 7: For Rows-Using numpy.square() method and [[]] operator.
df.loc[['a','b']] = np.square(df.loc[['a','b']])
df

#%%
#Example 8: Custome function in apply function
# creating a DataFrame
df = pd.DataFrame({'String 1' :['Tom', 'Nick', 'Krish', 'Jack'],
                   'String 2' :['Jane', 'John', 'Doe', 'Mohan']})
#df    
#print(df)
display(df)

def prepend_geek(name):
    return 'Geek ' + name
 
df[["String 1", "String 2"]] = df[["String 1", "String 2"]].apply(prepend_geek)
 
display(df)

#%%
#Example 2 : Multiplying the value of each element by 2 
df = pd.DataFrame({'Integers' :[1, 2, 3, 4, 5],
                   'Float' :[1.1, 2.2, 3.3, 4.4 ,5.5]})
display(df)
 
def multiply_by_2(number):
    return 2 * number
 
df[["Integers", "Float"]] = df[["Integers", "Float"]].apply(multiply_by_2)
 
display(df)

#%%
def squareData(x):
    return x * x
 
matrix = [(1,2,3,4),
          (5,6,7,8,),
          (9,10,11,12),
          (13,14,15,16)
         ]
 
df = pd.DataFrame(matrix, columns = list('abcd'))
new_df = df.apply(squareData)
new_df

#%%
new_df = df.apply(squareData, axis = 1)

#%%
# function to returns x+y
def addData(x, y):
    return x + y
 
matrix = [(1,2,3,4),
          (5,6,7,8,),
          (9,10,11,12),
          (13,14,15,16)
         ]
 
df = pd.DataFrame(matrix, columns = list('abcd'))
new_df = df.apply(addData, args = [1])
print(new_df)

#%%
new_df = df.apply(addData, axis = 1, args = [3])
new_df
 
#%%
#Return multiple columns using Pandas apply() method
    
#Objects passed to the pandas.apply() are Series objects whose 
#index is either the DataFrame’s index (axis=0) or the DataFrame’s
# columns (axis=1).

import pandas
import numpy
  
# Creating dataframe
df = pandas.DataFrame([[4, 9], ] * 3, columns =['A', 'B'])
print('Data Frame:')
display(df)

#%%  
# Using pandas.DataFrame.apply() on the whole data frame

print('Returning multiple columns from Pandas apply()')
df.apply(numpy.sqrt)    

#%%
dt = df.apply(np.sum) #get the sum of all values in each column(column wise)
dt

#%%
df.apply(numpy.sum, axis = 0) #np.square,np.sqrt etc

#%%
df.apply(numpy.sum, axis = 1) ##get the sum of all values in each row(row wise)


#%%
#%%
#%%
# List of Tuples
matrix = [(1, 2, 3),
          (4, 5, 6),
          (7, 8, 9)
         ]
  
# Create a DataFrame object
df = pd.DataFrame(matrix, columns = list('xyz'), index = list('abc'))  
print(df)

#%%
df['z1'] = np.square(df['x'])
df

#%%
df['z2'] = df['z'].apply(np.square)
df

#%%
df['z3'] = df.apply(np.sum, axis = 1)
df

#%%
def add(a, b, c):
    return a + b + c

df['add'] = df.apply(lambda row : add(row['x'],
                     row['y'], row['z']), axis = 1)

df

#%%
df.loc['adde'] = df.apply(lambda row : add(row['a'],
                     row['b'], row['c']), axis = 0)

df

#%%
matrix = [(1,2,3,4),
          (5,6,7,8,),
          (9,10,11,12),
          (13,14,15,16)
         ]
df = pd.DataFrame(matrix, columns = list('abcd'))
 
new_df = df.apply(lambda x : x + 10)
new_df

#%%
new_df = df.apply(lambda x: x + 5, axis = 1)

#%%
#Example #4: Generate range
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

# Function to generate range
def generate_range(n):
	# printing the range for eg:input is 67 output is 60-70
	n = int(n)
	lower_limit = n//10 * 10
	upper_limit = lower_limit + 10	
	return str(str(lower_limit) + '-' + str(upper_limit))
	
def replace(row):
	for i, item in enumerate(row):	
		# updating the value of the row
		row[i] = generate_range(item)
	return row
		

def main():
	# create a dictionary with three fields each
	data = {
			'A':[0, 2, 3],
			'B':[4, 15, 6],
			'C':[47, 8, 19] }
	
	df = pd.DataFrame(data)

	print('Before applying function: ')
	print(df)
	
	# applying function to each row in dataframe and storing result in a new column
	df = df.apply(lambda row : replace(row))
    
	print('After Applying Function: ')
	print(df)


#%%




































































































































#%%

import pandas as pd

# reading csv
s = pd.read_csv("stock.csv", squeeze = True)

# defining function to check price
def fun(num):

	if num<200:
		return "Low"

	elif num>= 200 and num<400:
		return "Normal"

	else:
		return "High"

# passing function to apply and storing returned series in new
new = s.apply(fun)

# printing first 3 element
print(new.head(3))

# printing elements somewhere near the middle of series
print(new[1400], new[1500], new[1600])

# printing last 3 elements
print(new.tail(3))


#%%

#adds 5 to each value in series and returns a new series.

import pandas as pd
s = pd.read_csv("stock.csv", squeeze = True)

# adding 5 to each value
new = s.apply(lambda num : num + 5)

# printing first 5 elements of old and new series
print(s.head(), '\n', new.head())

# printing last 5 elements of old and new series
print('\n\n', s.tail(), '\n', new.tail())

#%%
































