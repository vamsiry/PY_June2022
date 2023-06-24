# -*- coding: utf-8 -*-
"""
Created on Sat May  7 20:50:20 2022

@author: rvamsikrishna
"""
#%%
#%%
#********************1. reversed()*********************
# Reversed for string
seq_string = 'Python'
print(list(reversed(seq_string)))
#%%
# Reversed for tuple
seq_tuple = ('P', 'y', 't', 'h', 'o', 'n')
print(list(reversed(seq_tuple)))
#%%
# Reversed for range
seq_range = range(5, 9)
print(list(reversed(seq_range)))
#%%
# Reversed for list
seq_list = [1, 2, 4, 3, 5]
print(list(reversed(seq_list)))

#%%
#*******************************3. len()**************************
mylist = ["articles", "orange", "medium"]
x = len(mylist)
print(x)
#%%
myStr = "I am writing an article"
x = len(myStr)
print(x)
#%%   
#****************************4. isnumeric()**********************
#If you want to check if your string has only numeric values, this function 
#can be useful. This function is mostly used with regular expressions 
#validating any object.

s = '1242323'
print(s.isnumeric())
#%%
s='python12'
print(s.isnumeric())
#%%
s='15python'
print(s.isnumeric())

#%%
#%%
#******************************6. is_upper()*************************
#This function is used to check whether all the characters in the strings
# are in upper case.
txt = "THIS IS MeDIUM!"
x = txt.isupper()
print(x)

#%%
#%%
#**************************8. isnan()*********************************
#This function is very handy to check whether an object is NaN or not.
import math
# This function checks whether some values are NaN
print (math.isnan(56))
print (math.isnan(-45.34))
print (math.isnan(+45.34))
print (math.isnan(math.inf))
print (math.isnan(float("nan")))
print (math.isnan(float("inf")))
print (math.isnan(float("-inf")))
print (math.isnan(math.nan))

#%%
#%%
#**************************#Python abs()*********************************
number = -20
abs(number)

#%%
#%%
#**************************#Python all()*********************************
#The all() function returns True if all elements in the given iterable are true. 
#If not, it returns False.

#all() Syntax ---all(iterable)
#iterable - any iterable (list, tuple, dictionary, etc.) which contains the elements

#Example 1: Using all() on Python Lists
# all() method works in a similar way for tuples and sets like lists.

boolean_list = ['True', 'True', 'True']
result = all(boolean_list)
print(result)

#%%
# all values true
l = [1, 3, 4, 5]
print(all(l))
#%%
# all values false
l = [0, False]
print(all(l))
#%%
# one false value
l = [1, 3, 4, 0]
print(all(l))
#%%
# one true value
l = [0, False, 5]
print(all(l))
#%%
# empty iterable - Empty Iterable outputs TRUE and # 0 is False # '0' is True
l = []
print(all(l))
#%%
#Example 2: How all() works for strings?
s = "This is good"
print(all(s))
#%%
# 0 is False ---'0' is True
s = '000'
print(all(s))
#%%
s = ''
print(all(s))
#%%
#Example 3: How all() works with Python dictionaries?

#In the case of dictionaries, if all keys (not values) are true or the 
#dictionary is empty, all() returns True. Else, it returns false for all 
#other cases..
s = {0: 'False', 1: 'False'}
print(all(s))
#%%
s = {1: 'True', 2: 'True'}
print(all(s))
#%%
s = {1: 'True', False: 0}
print(all(s))
#%%
s = {}
print(all(s))
#%%
# 0 is False ---'0' is True
s = {'0': 'True'}
print(all(s))

#%%
#%%
#****************************Python any()**********************************

#The any() function returns True if any element of an iterable is True. 
#If not, it returns False.
#any() Syntax ---any(iterable)
#The any() function takes an iterable (list, string, dictionary etc.) in Python.

#Example 1: Using any() on Python Lists 
#The any() method works in a similar way for tuples and sets like lists.

boolean_list = ['True', 'False', 'True']
result = any(boolean_list)
print(result)

#%%
# True since 1,3 and 4 (at least one) is true
l = [1, 3, 4, 0]
print(any(l))
#%%
# False since both are False
l = [0, False]
print(any(l))
#%%
# True since 5 is true
l = [0, False, 5]
print(any(l))
#%%
# False since iterable is empty
l = []
print(any(l))
#%%
#Example 2: Using any() on Python Strings

# At east one (in fact all) elements are True
s = "This is good"
print(any(s))
#%%
# 0 is False --'0' is True since its a string character
s = '000'
print(any(s))
#%%
# False since empty iterable
s = ''
print(any(s))
#%%
#Example 3: Using any() with Python Dictionaries

#In the case of dictionaries, if all keys (not values) are false or the 
#dictionary is empty, any() returns False. If at least one key is true, 
#any() returns True.

# 0 is False
d = {0: 'False'}
print(any(d))
#%%
# 1 is True
d = {0: 'False', 1: 'True'}
print(any(d))
#%%
# 0 and False are false
d = {0: 'False', False: 0}
print(any(d))
#%%
# iterable is empty
d = {}
print(any(d))
#%%
# 0 is False --'0' is True
d = {'0': 'False'}
print(any(d))
#%%
#*******************Python compile()***************************************
#compile() method is used if the Python code is in string form or is an AST
# object, and you want to change it to a code object.

#The code object returned by compile() method can later be called using methods
# like: exec() and eval() which will execute dynamically generated Python code.


codeInString = 'a = 5\nb=6\nsum=a+b\nprint("sum =",sum)'
codeObejct = compile(codeInString, 'sumstring', 'exec')

exec(codeObejct)

#%%















