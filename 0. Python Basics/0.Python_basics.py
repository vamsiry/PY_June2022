# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:40:21 2019

@author: rvamsikrishna
"""
#%%
#=============================================================================================
#Lesson 1
#1. How to read and write in Python
#======================================

# print(), to output data from any Python program. 
print(5 + 10)
#%%
print(3 * 7, (17 - 2) * 8)
#%%
print(2 ** 4)  # two stars are used for exponentiation (2 to the power of 16)
#%%
print(37 / 3)  # single forward slash is a division
#%%
print(37 // 3)  # double forward slash is an integer division
# it returns only the quotient of the division (i.e. no remainder)
#%%
print(37 % 3)  # percent sign is a modulus operator
        # it gives the remainder of the left value divided by the right value

#%%
#To input data into a program, we use input(). 
#This function reads a single line of text, as a String.
        
print('What is your name?')
name = input()  # read a single line and store it in the variable "name"
print('Hi ' + name + '!')

print(f"Hi {name} !.")

#%%
# 2. Sum of numbers and strings
#======================================        
a = input()
b = input()
s = a + b
print(type(a))
print(a)
print(b)
print(s)

#%%        
#To cast (convert) the string of digits into an integer number, we can use the function int()
a = int(input())
b = int(input())
s = a + b
print(s)

#%%
first = 5
second = 7
print(first * second)
print(first + second)
print(first / second)
print(first // second)
#%%
# you can use single or double quotes to define a string
first = '5'
second = '7'
print(first * second)
print(first + second)

# if the two variables "first" and "second" are pointing to
#the objects of type int, Python can multiply them. However, 
#if they are pointing to the objects of type str, Python can't do that:

#%%
# Lesson 2 Integer and float numbers
#------------------------------------

print(17 / 3)  # gives 5.66666666667
print(2 ** 4)  # gives 16
print(2 ** -2) # gives 0.25
print(17 / 3)   # gives 5.66666666667
print(17 // 3)  # gives 5
print(17 % 3)   # gives 2

#%%
#2. Floating-point numbers
#When we read an integer value, we read a line with input() and then cast
# a string to integer using int(). When we read a floating-point number, 
#we need to cast the string to float using float():


x = float(input())
print(x)

print(int(1.3))   # gives 1
print(int(1.7))   # gives 1
print(int(-1.3))  # gives -1
print(int(-1.7))  # gives -1

print(round(1.3))   # gives 1
print(round(1.7))   # gives 2
print(round(-1.3))  # gives -1
print(round(-1.7))  # gives -2

print(0.1 + .3)  # gives 0.30000000000000004

#%%

#3. math module
#----------------

#Python has many auxiliary functions for calculations with floats. 
#They can be found in the math module. 

import math

x = math.ceil(4.2)
print(x)
print(math.ceil(1 + 3.8))

# or 
from math import ceil
 
x = 7 / 2
y = ceil(x)
print(y)

#%%

#Python Indentations
#--------------------
#Python uses indentation to indicate a block of code.

if 5 > 2:
  print("Five is greater than two!")
  
#%%
  
#Random Number
  
import random

print(random.randrange(1,10)) 

#%%
#https://www.w3schools.com/python/python_strings.asp

#Strings are Arrays

#Like many other popular programming languages, strings in Python are arrays
# of bytes representing unicode characters.
#python does not have a character data type, a single character is simply a 
#string with a length of 1. Square brackets can be used to access elements 
#of the string.

a = "Hello, World!"
print(a[1])


b = "Hello, World!"
print(b[2:5])


#The strip() method removes any whitespace from the beginning or the end:
a = " Hello, World! "
print(a.strip()) # returns "Hello, World!" 


#The len() method returns the length of a string:
a = "Hello, World!"
print(len(a))


#The lower() method returns the string in lower case:
a = "Hello, World!"
print(a.lower())


#The upper() method returns the string in upper case:
a = "Hello, World!"
print(a.upper())


#The replace() method replaces a string with another string:
a = "Hello, World!"
print(a.replace("H", "J"))


#The split() method splits the string into substrings if it finds 
#instances of the separator:
a = "Hello, World!"
print(a.split(",")) # returns ['Hello', ' World!'] 

#%%
#we cannot combine strings and numbers like this:
age = 36
#txt = "My name is John, I am " + age
#print(txt)
print(f"My name is John, I am {age} .")

#%%
#But we can combine strings and numbers by using the format() method!
#The format() method takes the passed arguments, formats them, and places 
#them in the string where the placeholders {} are:
age = 36
txt = "My name is John, and I am {}"
print(txt.format(age))

#%%
#The format() method takes unlimited number of arguments, and 
#are placed into the respective placeholders:
quantity = 3
itemno = 567
price = 49.95
myorder = "I want {} pieces of item {} for {} dollars."
print(myorder.format(quantity, itemno, price))

#%%

#You can use index numbers {0} to be sure the arguments are placed in the correct placeholders:
quantity = 3
itemno = 567
price = 49.95
myorder = "I want to pay {2} dollars for {0} pieces of item {1}."
print(myorder.format(quantity, itemno, price))


#%%
























