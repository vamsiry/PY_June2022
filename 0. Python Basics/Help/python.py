# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:06:25 2018

@author: vamsi
"""
#%%
import pandas as pd
import numpy as np

#%%
df = pd.DataFrame ({'a' : np.random.randn(6),
             'b' : ['foo', 'bar'] * 3,
             'c' : np.random.randn(6)})

#%%
def my_test(a, b):
    return a % b

df['Value'] = df.apply(lambda row: my_test(row['a'], row['c']), axis=1)
#%%
df
df.shape
type(df)
#%%

def my_test2(row):
   return row['a'] % row['c']

df['Value'] = df.apply(my_test2, axis=1)

df
#%%
#%%
# Source : https://chrisalbon.com/python/data_wrangling/pandas_apply_operations_to_dataframes/
import pandas as pd
import numpy as np
data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'year': [2012, 2012, 2013, 2014, 2014], 
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
df
#%%
capitalizer = lambda x: x.upper()

df['name'].apply(capitalizer) #apply() can apply a function along any axis of the dataframe

#%%
df['name'].map(capitalizer) #map() applies an operation over each element of a series

#%%
# Drop the string variable so that applymap() can run
df = df.drop('name', axis=1)

#%%
#applymap() applies a function to every single element in the entire dataframe.
# Return the square root of every cell in the dataframe
df.applymap(np.sqrt)

#%%
#Create a function that multiplies all non-strings by 100
def times100(x):
    # that, if x is a string,
    if type(x) is str:
        # just returns it untouched
        return x
    # but, if not, return it multiplied by 100
    elif x:
        return 100 * x
    # and leave everything else
    else:
        return
#%%
df.applymap(times100)
#%%
#%%
# Source : https://www.python-course.eu/python3_lambda.php
sum = lambda x, y : x + y

sum(3,4)

#%%
def sum(x,y):
 return x + y

sum(3,4)
#%%
#the advantage of the lambda operator can be seen when it is used in combination with the map() function
#r = map(func, seq)
#first argument func is the name of a function and the second a sequence (e.g. a list) seq.
# map() applies the function func to all the elements of the sequence seq.

# map() used to return a list, where each element of the result list was the result
# of the function func applied on the corresponding element of the list or tuple
# "seq". With Python 3, map() returns an iterator. 

def fahrenheit(T):
     return ((float(9)/5)*T + 32)
 
def celsius(T):
     return (float(5)/9)*(T-32)
#%% 
temperatures = (36.5, 37, 37.5, 38, 39)
F = map(fahrenheit, temperatures)
C = map(celsius, F)


#%%
temperatures_in_Fahrenheit = list(map(fahrenheit, temperatures))
#%%
temperatures_in_Celsius = list(map(celsius, temperatures_in_Fahrenheit))
#%%
print(temperatures_in_Fahrenheit)
#%%
print(temperatures_in_Celsius)
#%% 
# above we haven't used lambda. By using lambda, we wouldn't have had to 
#define and name the functions fahrenheit() and celsius()

C = [39.2, 36.5, 37.3, 38, 37.8] 

F = list(map(lambda x: (float(9)/5)*x + 32, C))

print(F)

C = list(map(lambda x: (float(5)/9)*(x-32), F))

print(C)

#%%
#map() can be applied to more than one list. The lists have to have the same length.

a = [1,2,3,4]
b = [17,12,11,10]
c = [-1,-4,5,9]

list(map(lambda x,y:x+y, a,b))

list(map(lambda x,y,z:x+y+z, a,b,c))

list(map(lambda x,y,z : 2.5*x + 2*y - z, a,b,c))

#%%
#to do encoding and decoding in Python, we can use ord('a') and chr(97) 
list(map(chr,[66,53,0,94]))

#%% 
# https://www.programiz.com/python-programming/methods/built-in/any
#Example 1: How any() works with Python List?
l = [1, 3, 4, 0]
print(any(l))

l = [0, False]
print(any(l))

l = [0, False, 5]
print(any(l))

l = []
print(any(l))

#Example 2: How any() works with Python Strings?
s = "This is good"
print(any(s))

# 0 is False
# '0' is True
s = '000'
print(any(s))

s = ''
print(any(s))

#Example 3: How any() works with Python Dictionaries?
d = {0: 'False'}
print(any(d))

d = {0: 'False', 1: 'True'}
print(any(d))

d = {0: 'False', False: 0}
print(any(d))

d = {}
print(any(d))

# 0 is False
# '0' is True
d = {'0': 'False'}
print(any(d))


#%%











