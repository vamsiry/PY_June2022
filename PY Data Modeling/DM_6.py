# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:38:14 2022

@author: rvamsikrishna
"""
#https://www.geeksforgeeks.org/iterating-over-rows-and-columns-in-pandas-dataframe/?ref=lbp


#File-Contents
#--------------

#1. Different ways to iterate over rows in  Dataframe
#2. Different ways to iterate over columns in  Dataframe
#3. Different ways to iterate over all or certain columns in  Dataframe


# Iterate over the rows
#***************************
#Method #1 : Using index attribute of the Dataframe .
#Method #2 : Using loc[] function of the Dataframe.
#Method #3 : Using iloc[] function of the DataFrame.

#Method #4 : Using iterrows() method of the Dataframe.
# iterrows() function returns each index value along with a series 
#containing the data in each row. 

#Method #6 : Using iteritems() method of the Dataframe.
#iteritems() iterates over each column as key, value pair with the label 
#as key, and column value as a Series object.

#Method #5 : Using itertuples() method of the Dataframe.
# itertuples() return a tuple for each row in the DataFrame. 
#The first element of the tuple will be the row’s corresponding index value, 
#while the remaining values are the row values"

#Method #6 : Using apply() method of the Dataframe.

#Method #7 :df.items()

# Iterate over the Columns
#**************************



#%%
##Different ways to iterate over rows in Pandas Dataframe
#------------------------------------------------------------
#Iteration is a general term for taking each item of something, one after another. 

#Pandas DataFrame consists of rows and columns so, in order to iterate over 
#dataframe, we have to iterate a dataframe like a dictionary.

#In a dictionary, we iterate over the keys of the object in the same way 
#we have to iterate in dataframe


# import pandas package as pd
import pandas as pd

#Define a dictionary containing students data
data = {'Name': ['Ankit', 'Amit', 'Aishwarya', 'Priyanka'],
				'Age': [21, 19, 20, 18],
				'Stream': ['Math', 'Commerce', 'Arts', 'Biology'],
				'Percentage': [88, 92, 95, 70]}

# Convert the dictionary into DataFrame
df = pd.DataFrame(data, columns = ['Name', 'Age', 'Stream', 'Percentage'])

print("Given Dataframe :\n", df)

print("\nIterating over rows using index attribute :\n")

#%%
#%%
# Iterate over the rows
#****************************
#Method #1 : Using index attribute of the Dataframe .

# iterate through each row and select 'Name' and 'Stream' column respectively.

for ind in df.index:
	print(df['Name'][ind], " ", df['Stream'][ind])

#%%
#%%    
#Method #2 : Using loc[] function of the Dataframe.
    
for i in range(len(df)) :
  print(df.loc[i, "Name"], df.loc[i, "Age"])

#%%    
#Method #3 : Using iloc[] function of the DataFrame.
  
for i in range(len(df)) :
  print(df.iloc[i, 0], df.iloc[i, 2])

#%%
#%%  
#Method #4 : Using iterrows() method of the Dataframe.
  
# iterrows() function returns each index value along with a series 
#containing the data in each row. 

for index, row in df.iterrows():
    print (row["Name"], row["Age"])
    
#%%
for index, row in df.iterrows():
    print (row.Name, row.Age)    
    
#%%
for i, j in df.iterrows():
    print(i, j)
    print()
    
#%%
import pandas as pd
    
print(df.iterrows()) 
 
print(pd.Series(df.iterrows()))
  
#print(list(df.iterrows()))
   
print(type((df.iterrows())))

print(type(list(df.iterrows())))

#%%    
#%%  
#Method #6 : Using iteritems() method of the Dataframe.

#iteritems() iterates over each column as key, value pair with the label 
#as key, and column value as a Series object.

for key, value in df.iteritems():
    print(key, value)
    print()
    
    
#%%
import pandas as pd    

print(df.iteritems()) 
   
print(pd.Series(df.iteritems()))

#print(list(df.iteritems()))
   
print(type((df.iteritems())))

print(type(list(df.iteritems())))


#%%
#%%  
#Method #5 : Using itertuples() method of the Dataframe.

# itertuples() return a tuple for each row in the DataFrame. 
#The first element of the tuple will be the row’s corresponding index value, 
#while the remaining values are the row values.

for row in df.itertuples(index = True, name ='Pandas'):
    print (getattr(row, "Name"), getattr(row, "Percentage"))
    
#%%
for row in df.itertuples():
    print(row)
    
#%%
for row in df.itertuples(index=False):
    print(row)

#%%
#With the name parameter set we set a custom name for the yielded named tuples:
for row in df.itertuples(name='Animal'):
    print(row)      

#%%    
for row in df.itertuples():
    print(row[1:5])

#%%
#  capture the row number while iterating: 
for row in df.itertuples():
    print(row.Index, row.Name) 
    
#%%
# capture the row number while iterating:    
for i, row in enumerate(df.itertuples(), 1):
    print(i, row.Name)    
    
    
#%%
print(df.itertuples()) 

print(tuple(df.itertuples()))
   
#print(list(df.itertuples()))
   
print(type((df.itertuples())))

print(type(list(df.itertuples())))



#%%
#Method #6 : Using apply() method of the Dataframe.
    
print(df.apply(lambda row: row["Name"] + " " + str(row["Percentage"]), axis = 1))

#%%    
#%%
# Iterate over the Columns
#***************************
#In order to iterate over columns, we need to create a list of 
#dataframe columns and then iterating through that list to pull
# out the dataframe columns.

# creating a list of dataframe columns
columns = list(df)

for i in columns:
	# printing the third element of the column
	print (df[i][2])

#%%
#%%
#Loop or Iterate over all or certain columns of a dataframe     
#**********************************************************

#Method #1: Using DataFrame.iteritems(): 
#---    
#every column in the Dataframe it returns an iterator to the tuple containing 
#the column name and its contents as series.
 
for (columnName, columnData) in df.iteritems():
    print('Column Name : ', columnName)
    print('Column Contents : ', columnData.values)    
    
#%%
#Method #2: Using [ ] operator : 

for column in df:
    columnSeriesObj = df[column]
    print('Column Name : ', column)
    print('Column Contents : ', columnSeriesObj.values)    

#%%
#Method #3: Iterate over more than one column : 
for column in df[['Name', 'Age']]:
    columnSeriesObj = df[column]
    print('Column Name : ', column)
    print('Column Contents : ', columnSeriesObj.values)    

#%%
#Method #4: Iterating columns in reverse order : 
for column in reversed(df.columns):
    columnSeriesObj = df[column]
    print('Column Name : ', column)
    print('Column Contents : ', columnSeriesObj.values)    


#%%
#Method 7: df.items
#--------------------
#Iterates over the DataFrame columns, returning a tuple with the
# column name and the content as a Series.
    
    
df = pd.DataFrame({'species': ['bear', 'bear', 'marsupial'],
                  'population': [1864, 22000, 80000]},
                  index=['panda', 'polar', 'koala'])    


for label, content in df.items():
    print(f'label: {label}')
    print(f'content: {content}', sep='\n')

#%%




