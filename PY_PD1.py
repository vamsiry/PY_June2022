# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:46:47 2022

@author: rvamsikrishna
"""

#File-Contents
#--------------
# 1. What is pandas Series and ways to Create series 
# 2. What is pandas DataFrame and ways to Create Data Frame 
# 3. Creating a nested list for wach row from the data frame

#%%
#What is a Series?
#----------------
#1. A Pandas Series is like a column in a table.

#2. It is a one-dimensional array holding data of any type.(integer, float, and string)

#3. Labels/Index -- If nothing else is specified, the values are labeled with 
#their index number.  First value has index 0, second value has 
#index 1 etc. This label can be used to access a specified value.

#A series, by definition, cannot have multiple columns.

#Data: can be a list, dictionary or scalar value 

dir(pd.core.series.Series) #to get a list of available methods.

#%%
import pandas as pd
a = [1, 7, 2]
myvar = pd.Series(a)
print(myvar)

#%%
print(myvar[0])
#%%
#Create Labels -- With the "index" argument, you can name your own labels.
myvar = pd.Series(a, index = ["x", "y", "z"])
print(myvar["y"])

#%%
#You can also use a key/value object, like a dictionary, when creating a 
#Series.  --  Note: The keys of the dictionary become the labels.
calories = {"day1": 420, "day2": 380, "day3":   390}
myvar = pd.Series(calories)
print(type(myvar))
print(myvar['day1'])

#%%
#To select only some of the items in the dictionary, use the index argument
# and specify only the items you want to include in the Series.
    
#Create a Series using only data from "day1" and "day2":
calories = {"day1": 420, "day2": 380, "day3":   390, "day4": 390,}
myvar = pd.Series(calories,   index = ["day1", "day2"])
print(myvar)




#%%
#%%
#%%
#What is a data frame?
#----------------------
#A Pandas DataFrame is a 2 dimensional data structure, like a 2 dimensional 
#array, or a table with rows and columns.

#Labels/Index -- If nothing else is specified, the values are labeled with their 
#index number.  First value has index 0, second value has index 1 etc. 
#This label can be used to access a specified value.

##Pandas DataFrame can be created from the arrays, lists, dictionary, and 
#from a list of dictionary etc.

#You can get a list of available DataFrame methods using the Python dir function:
dir(pd.DataFrame)

#And you can get the description of each method using help:
help(pd.DataFrame.mean)

#%%
#Example 1: #Creates a DataFrame using np arrays and adding indexes
#-------------------------------------------------------------------
import numpy as np
import pandas as pd

my_array = np.array([[11,22,33],[44,55,66]])

df = pd.DataFrame(my_array, columns = ['Column_A','Column_B','Column_C'])

#df = pd.DataFrame(my_array, columns = ['Column_A','Column_B','Column_C'],index = ['Item_1', 'Item_2'])

print(type(my_array))
print(type(df))
print(my_array)
print(df)

#%%
#array with multiple data types to convert to data frame with index

my_array = np.array([['Jon',25,1995,2016],['Maria',47,1973,2000],['Bill',38,1982,2005]], dtype=object)

df = pd.DataFrame(my_array, columns = ['Name','Age','Birth Year','Graduation Year'])

print(type(df))
print(df.dtypes)
print(df)

#%%
df['Age'] = df['Age'].astype(int)
df['Birth Year'] = df['Birth Year'].astype(int)
df['Graduation Year'] = df['Graduation Year'].astype(int)

print(type(df))
print(df.dtypes)
print(df)

#%%
#pandas data frame to numpy array
ary = np.array(df)
print(type(ary))
print(ary)



#%%
#%%
#Example 2: #Creates a DataFrame using Dictionary. and adding indexes
#----------------------------------------------------------------
dic = {'name':["aparna", "pankaj", "sudhir", "Geeku"], 
        'degree': ["MBA", "BCA", "M.Tech", "MBA"], 
        'score':[90, 40, 80, 98]} 
  
df = pd.DataFrame(dict, index = ['A', 'B', 'C', 'D']) 

print(type(dic))
print(type(df))
print(dic)
print(df) 

#%%
print(df.loc['A'])  #accessing row of data frame using index
print(df.loc[['A','B']])  #accessing row of data frame using index

#%%
#%%
#Example 3: Create pandas dataframe from lists using dictionary:
#----------------------------------------------------------------------------
# Using lists in dictionary to create dataframe
nme = ["aparna", "pankaj", "sudhir", "Geeku"] 
deg = ["MBA", "BCA", "M.Tech", "MBA"] 
scr = [90, 40, 80, 98] 
  
# dictionary of lists  
dict = {'Name': nme, 'Degree': deg, 'Score': scr}      

df = pd.DataFrame(dict)  
df.index += 1   

print(type(df))
print(df)


#%%
#Example 4: Creating DataFrame using multi-dimensional list/lists of lists.
#--------------------------------------------------------------------

lst = [['tom', 'reacher', 25], ['krish', 'pete', 30], 
       ['nick', 'wilson', 26], ['juli', 'williams', 22]]

print(type(lst))   
#lst.shape throws error
#Use numpy.array to use shape attribute.
#import numpy as np    
#np.shape(lst)

df = pd.DataFrame(lst, columns =['F_Name', 'L_Name', 'Age'], dtype = float) 

print(type(df))
print(df) 


#%%
#%%
#Example 5: create pandas DataFrame from lists of dictionaries 
#--------------------------------------------------------
# Initialise data to lists. 
data = [{'a': 1, 'b': 2, 'c':3}, {'a':10, 'b': 20, 'c': 30}] 
df = pd.DataFrame(data) 
print(type(df)) 
print(df)


#%%
#%%
##Creating DataFrame using zip() function.(list of tuples)
#----------------------------------------------------------
# List1  
Name = ['tom', 'krish', 'nick', 'juli']      
# List2  
Age = [25, 30, 26, 22]      

# get the list of tuples from two lists, and merge them by using zip().  
list_of_tuples = list(zip(Name, Age))  
    
# Assign data to tuples.  
print(type(list_of_tuples))
print(list_of_tuples)

#%%
df = pd.DataFrame(list_of_tuples, columns = ['Name', 'Age'])   
df  


#%%
#%%
#%%
#Create list of list for each row from Pandas DataFrame 
#--------------------------------------------------------------
df = pd.DataFrame({'Date':['10/2/2011', '11/2/2011', '12/2/2011', '13/2/11'], 
                    'Event':['Music', 'Poetry', 'Theatre', 'Comedy'], 
                    'Cost':[10000, 5000, 15000, 2000]}) 

df.shape
    
#%%
Row_list =[] 
# Iterate over each row 
for i in range((df.shape[0])):   
    # Using iloc to access the values of the current row denoted by "i" 
    Row_list.append(list(df.iloc[i, :])) 
  
# Print the list 
print(Row_list) 
print(len(Row_list))
#print(Row_list[:3]) 

#%%
#Now we will use the DataFrame.iterrows() function to iterate 
#over each of the row of the given Dataframe and construct a
# list out of the data of each row.

#Just like any other Python’s list we can perform any list operation 
#on the extracted list.

# Create an empty list 
Row_list =[] 
# Iterate over each row 
for index, rows in df.iterrows(): 
    my_list =[rows.Date, rows.Event, rows.Cost] 
    Row_list.append(my_list) 
  
print(Row_list) 

#%%
# we can use DataFrame.itertuples() function and then we can
# append the data of each row to the end of the list.

Row_list =[] 
# Iterate over each row 
for rows in df.itertuples(): 
    my_list =[rows.Date, rows.Event, rows.Cost] 
    Row_list.append(my_list) 
  
print(Row_list) 


#%%
#%%
#%%

#Creating Multi index data frame
#-------------------------------
idx = pd.MultiIndex.from_product([['Zara', 'LV', 'Roots'],
                                  ['Orders', 'GMV', 'AOV']],
                                 names=['Brand', 'Metric'])
col = ['Yesterday', 'Yesterday-1', 'Yesterday-7', 'Thirty day average']

df = pd.DataFrame('-', idx, col)
df

#%%
#Creating data frame using random num generator
#-------------------------------------------------------
import pandas as pd
import numpy as np
np.random.seed(5)
df = pd.DataFrame(np.random.randint(100, size=(100, 6)), 
                  columns=list('ABCDEF'), 
                  index=['R{}'.format(i) for i in range(100)])
df.head()

#%%































