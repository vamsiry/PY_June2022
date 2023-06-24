# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:27:53 2022

@author: rvamsikrishna
"""

#categorical data handling
#https://www.datacamp.com/community/tutorials/categorical-data
import pandas as pd

lst = [1, 2, 3, 1, 2, 3]
s = pd.Series([1, 2, 3, 10, 20, 30], lst)
grouped = s.groupby(level=0)
grouped.first()
grouped.last()
grouped.sum()

#%%
df3 = pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]})
df3.groupby(["X"]).get_group("A")
df3.groupby(["X"]).get_group("B")
#%%
df3.groupby("X").groups


#%%
##GroupBy dropna
df_list = [[1, 2, 3], [1, None, 4], [2, 1, 3], [1, 2, 2]]
dff = pd.DataFrame(df_list, columns=["a", "b", "c"])

dff.groupby(by=["b"], dropna=True).sum()

dff.groupby(by=["b"], dropna=False).sum()

#%%

#(3) Reshaping DataFrames
#---------------------------
    
import pandas as pd

players_data = {'Player': ['Superman', 'Batman', 'Thanos', 'Batman', 'Thanos',
   'Superman', 'Batman', 'Thanos', 'Black Widow', 'Batman', 'Thanos', 'Superman'],
   'Year': [2000,2000,2000,2001,2001,2002,2002,2002,2003,2004,2004,2005],
   'Points':[23,43,45,65,76,34,23,78,89,76,92,87]}
   
df = pd.DataFrame(players_data)

print(df)

#%%
##Transpose

import pandas as pd

players_data = {'Player': ['Superman', 'Batman', 'Thanos', 'Batman', 'Thanos',
   'Superman', 'Batman', 'Thanos', 'Black Widow', 'Batman', 'Thanos', 'Superman'],
   'Year': [2000,2000,2000,2001,2001,2002,2002,2002,2003,2004,2004,2005],
   'Points':[23,43,45,65,76,34,23,78,89,76,92,87]}
   
df = pd.DataFrame(players_data)

print(df)

#Groupby
groups_df = df.groupby('Player')

for player, group in groups_df:
   print("----- {} -----".format(player))
   print(group)
   print("")
    
# Stacking  
df = df.stack()
print(df)

#%%
#%%
numbers = (1, 2, 3, 4)
result = map(lambda x: x + x, numbers)
print(list(result))

#%%
numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]
result = map(lambda x, y: x + y, numbers1, numbers2)
print(list(result))


#%%

# Return double of n
def addition(n):
    return n + n
 
# We double all numbers using map()
numbers = (1, 2, 3, 4)
result = map(addition, numbers)
print(list(result))

#%%
#Filter	The filter() function is used to generate an output list of values that return true when the function is called.
y = filter(lambda x: (x>=3), (1,2,3,4))
print(list(y))

#%%

#Reduce Function
#The reduce() function applies a provided function to ‘iterables’ and returns
# a single value, as the name implies.

#The function specifies which expression should be applied to the 
#‘iterables’ in this case. The function tools module must be 
#used to import this function.

reduce(lambda a,b: a+b,[23,21,45,98])


#%%
def factorial(x):
    if x == 1:
        return 1
    else:
        return (x * factorial(x-1))

factorial(num)

#%%

#return vowel or consonent
def get_letter_type(letter):
    if letter.lower() in 'aeiou':
        return 'vowel'
    else:
        return 'consonant'

grouped = df.groupby(get_letter_type, axis=1)

           
#%%           
#prime num in oython 
for num in range(1, 20):
   if num > 1:
       for i in range(2, num):
           if (num % i) == 0:
               break
       else:
           print(num)

#%%
#%%
lower = 2
upper = 20

print("Prime numbers between", lower, "and", upper, "are:")

for num in range(lower, upper + 1):
   if num > 1:
       for i in range(2, num):
           if (num % i) == 0:
               break
       else:
           print(num)

#%%
# Program to check if a number is prime or not

def isprime(num):
    for n in range(2,int(num**0.5)+1):
        if num%n==0:
            return False
    return True


print(isprime(7))
print(isprime(8))

#%%










