# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:27:06 2019

@author: rvamsikrishna
"""

#Number Data Type in Python
#----------------------------
#Python supports integers, floating point numbers and complex numbers. 
#They are defined as int, float and complex class in Python.

#Integers and floating points are separated by the presence or absence of a decimal point. 
#5 is integer whereas 5.0 is a floating point number.

#Complex numbers are written in the form, x + yj, where x is the real part and
# y is the imaginary part.

#We can use the type() function to know which class a variable or a value belongs 
# to and isinstance() function to check if it belongs to a particular class.

a = 5

# Output: <class 'int'>
print(type(a))

# Output: <class 'float'>
print(type(5.0))

# Output: (8+3j)
c = 5 + 3j
print(c + 3)

# Output: True
print(isinstance(c, complex))
#%%

#While integers can be of any length, a floating point number is accurate only up
#to 15 decimal places (the 16th place is inaccurate).

#Numbers we deal with everyday are decimal (base 10) number system. But computer 
#programmers (generally embedded programmer) need to work with binary (base 2),
# hexadecimal (base 16) and octal (base 8) number systems.

#In Python, we can represent these numbers by appropriately placing a prefix before that number.

#Binary 	'0b' or '0B'
#Octal 	'0o' or '0O'
#Hexadecimal 	'0x' or '0X'

# Output: 107
print(0b1101011)

# Output: 253 (251 + 2)
print(0xFB + 0b10)

# Output: 13
print(0o15)

#%%
#Type Conversion
#--------------------

#We can convert one type of number into another. This is also known as coercion.

#Operations like addition, subtraction coerce integer to float implicitly (automatically), 
#if one of the operand is float.

1 + 2.0

#%%
#We can also use built-in functions like int(), float() and complex() 
#to convert between types explicitly.

int(2.3)
int(-2.8)
float(5)
complex('3+5j')

#%%
#Python Decimal
#=================

#Python built-in class float performs some calculations that might amaze us. 
#We all know that the sum of 1.1 and 2.2 is 3.3, but Python seems to disagree.
(1.1 + 2.2) == 3.3

#It turns out that floating-point numbers are implemented in computer hardware as
# binary fractions, as computer only understands binary (0 and 1). Due to this reason, 
#most of the decimal fractions we know, cannot be accurately stored in our computer.

#Let's take an example. We cannot represent the fraction 1/3 as a decimal number.
# This will give 0.33333333... which is infinitely long, and we can only approximate it.

#Turns out decimal fraction 0.1 will result into an infinitely long binary fraction 
#of 0.000110011001100110011... and our computer only stores a finite number of it.

#This will only approximate 0.1 but never be equal. Hence, it is the limitation 
#of our computer hardware and not an error in Python.

1.1 + 2.2

#%%
#To overcome this issue, we can use decimal module that comes with Python. 
#While floating point numbers have precision up to 15 decimal places, 
#the decimal module has user settable precision.

import decimal

# Output: 0.1
print(0.1)

# Output: Decimal('0.1000000000000000055511151231257827021181583404541015625')
print(decimal.Decimal(0.1))

#It also preserves significance. We know 25.50 kg is more accurate than 25.5 kg as
# it has two significant decimal places compared to one.

from decimal import Decimal as D
# Output: Decimal('3.3')
print(D('1.1') + D('2.2'))

# Output: Decimal('3.000')
print(D('1.2') * D('2.50'))


#We might ask, why not implement Decimal every time, instead of float? The main 
#reason is efficiency. Floating point operations are carried out must faster than
# Decimal operations.

#%%
#When to use Decimal instead of float?
#-------------------------------------------
#When we are making financial applications that need exact decimal representation.
#When we want to control the level of precision required.
#When we want to implement the notion of significant decimal places.
#When we want the operations to be carried out like we did at school

#%%
#Python Fractions
#=====================
#A fraction has a numerator and a denominator, both of which are integers. 
#This module has support for rational number arithmetic.

import fractions

# Output: 3/2
print(fractions.Fraction(1.5))

# Output: 5
print(fractions.Fraction(5))

# Output: 1/3
print(fractions.Fraction(1,3))

#While creating Fraction from float, we might get some unusual results.
#This is due to the imperfect binary floating point number representation 
#as discussed in the previous section.

#Fortunately, Fraction allows us to instantiate with string as well.
# This is the preferred options when using decimal numbers.
#%%
import fractions

# As float
# Output: 2476979795053773/2251799813685248
print(fractions.Fraction(1.1))

# As string
# Output: 11/10
print(fractions.Fraction('1.1'))

#This datatype supports all basic operations. Here are few examples.
from fractions import Fraction as F

# Output: 2/3
print(F(1,3) + F(1,3))

# Output: 6/5
print(1 / F(5,6))

# Output: False
print(F(-3,10) > 0)

# Output: True
print(F(-3,10) < 0)

#%%
#Python Mathematics
#========================

# Math Module : https://www.programiz.com/python-programming/modules/math

#Python offers modules like math and random to carry out different mathematics 
#like trigonometry, logarithms, probability and statistics, etc.

import math

# Output: 3.141592653589793
print(math.pi)

# Output: -1.0
print(math.cos(math.pi))

# Output: 22026.465794806718
print(math.exp(10))

# Output: 3.0
print(math.log10(1000))

# Output: 1.1752011936438014
print(math.sinh(1))

# Output: 720
print(math.factorial(6))

#%%
import random

# Output: 16
print(random.randrange(10,20))

x = ['a', 'b', 'c', 'd', 'e']

# Get random choice
print(random.choice(x))

# Shuffle x
random.shuffle(x)

# Print the shuffled x
print(x)

# Print random element
print(random.random())

#%%
#%%
#%%
#Python Strings
#===============
#A string is a sequence of characters.
#conversion of character to a number is called encoding, and the reverse process is decoding.
#ASCII and Unicode are some of the popular encoding's are in use
#In Python, string is a sequence of Unicode character. Unicode was introduced
# to include every character in all languages and bring uniformity in encoding

#How to create a string in Python?

#Strings can be created by enclosing characters inside a single quote or double quotes.
#Even triple quotes can be used in Python but generally used to represent multiline strings and docstrings.

# all of the following are equivalent
my_string = 'Hello'
print(my_string)

my_string = "Hello"
print(my_string)

my_string = '''Hello'''
print(my_string)

# triple quotes string can extend multiple lines
my_string = """Hello, welcome to
           the world of Python"""
print(my_string)

#%%
#How to access characters in a string?
#------------------------------------------

str = 'programiz'
print('str = ', str)

#first character
print('str[0] = ', str[0])

#last character
print('str[-1] = ', str[-1])

#slicing 2nd to 5th character
print('str[1:5] = ', str[1:5])

#slicing 6th to 2nd last character
print('str[5:-2] = ', str[5:-2])

#%%
#If we try to access index out of the range or use decimal number, we will get errors.
str[15] 

str[1.5] 

#%%
#%%
#How to change or delete a string?
#-------------------------------------

#Strings are immutable. This means that elements of a string cannot be changed 
#once it has been assigned. We can simply reassign different strings to the same name.

my_string = 'programiz'
my_string[5] = 'a'

#%%
my_string = 'Python' # reassigning different string to the same variable
my_string

#%%
#We cannot delete or remove characters from a string. But deleting the string
#entirely is possible using the keyword del.

del my_string[1]

#%%
del my_string

my_string # throw error bcz already used del function with it

#%%
#Python String Operations
#------------------------------

#There are many operations that can be performed with string which 
#makes it one of the most used datatypes in Python.

#Concatenation of Two or More Strings
#----------------------------------------

# + operator does this in Python
# * operator can be used to repeat the string for a given number of times.

str1 = 'Hello'
str2 ='World!'

# using +
print('str1 + str2 = ', str1 + str2)

# using *
print('str1 * 3 =', str1 * 3)

#%%

#Writing two string literals together also concatenates them like + operator.
'Hello ' 'World!'

#If we want to concatenate strings in different lines, we can use parentheses.

s = ('Hello '
     'World '
     'vamsi')

s

#%%
#Iterating Through String
#------------------------------
#Using for loop we can iterate through a string.

count = 0

for letter in 'Hello World':
    if(letter == 'l'):
        count += 1

print(count,'letters found')

#%%
#%%
#String Membership Test
#--------------------------
'a' in 'program'
#%%
'at' not in 'battle'

#%%
#%%
#Built-in functions to Work with Python
#-------------------------------------------

#Various built-in functions that work with sequence, works with string as well.
#Some of the commonly used ones are enumerate() and len()

#The enumerate() function returns an enumerate object. 
#It contains the index and value of all the items in the string as pairs.
#This can be useful for iteration.

#len() returns the length (number of characters) of the string.

str = 'cold'

# enumerate()
list_enumerate = list(enumerate(str))
print('list(enumerate(str) = ', list_enumerate)

#%%
#character count
print('len(str) = ', len(str))

#%%
#Python String Formatting
#---------------------------

#Escape Sequence
#----------------

#If we want to print a text like -He said, "What's there?"- we can neither 
#use single quote or double quotes. This will result into SyntaxError as 
#the text itself contains both single and double quotes.
    
# print("He said, "What's there?"") #SyntaxError: invalid syntax

# print('He said, "What's there?"') #SyntaxError: invalid syntax

# One way to get around this problem is to use triple quotes. 
# Alternatively, we can use escape sequences.
# An escape sequence starts with a backslash and is interpreted differently.
#If we use single quote to represent a string, all the single quotes inside
# the string must be escaped
#Similar is the case with double quotes

# using triple quotes
print('''He said, "What's there?"''')

print('''He said, What's there?''')

# escaping single quotes
print('He said, "What\'s there?"')

# escaping double quotes
print("He said, \"What's there?\"")

print("He said, What\'s there?")

print("""He said, "What's there? """)

print("C:\\Python32\\Lib")

print("This is printed\nin two lines")

print("This is \x48\x45\x58 representation")

#%%
#Raw String to ignore escape sequence
#---------------------------------------

#Sometimes we may wish to ignore the escape sequences inside a string. 
#To do this we can place r or R in front of the string.
#This will imply that it is a raw string and any escape sequence inside it will be ignored.

print("This is \x61 \ngood example")

print(r"This is \x61 \ngood example")

#%%
#%%
#The format() Method for Formatting Strings
#--------------------------------------------

#Format strings contains curly braces {} as placeholders or replacement 
# fields which gets replaced.

#We can use positional arguments or keyword arguments to specify the order.

# default(implicit) order
default_order = "{}, {} and {}".format('John','Bill','Sean')
print('\n--- Default Order ---')
print(default_order)

# order using positional argument
positional_order = "{1}, {0} and {2}".format('John','Bill','Sean')
print('\n--- Positional Order ---')
print(positional_order)

# order using keyword argument
keyword_order = "{s}, {b} and {j}".format(j='John',b='Bill',s='Sean')
print('\n--- Keyword Order ---')
print(keyword_order)

#%%
#The format() method can have optional format specifications. 
#They are separated from field name using colon. For example, we can left-justify <, 
#right-justify > or center ^ a string in the given space. We can also format integers
#as binary, hexadecimal etc. and floats can be rounded or displayed in the 
#exponent format. There are a ton of formatting you can use. Visit here for all
#the string formatting available with the format() method.

## formatting integers
"Binary representation of {0} is {0:b}".format(12)
#%%
print("Binary representation of {0} is {0:b}".format(12))
#%%
# formatting floats
"Exponent representation: {0:e}".format(1566.345)
#%%
# round off
"One third is: {0:.3f}".format(1/3)
#%%
# string alignment
"|{:<10}|{:^10}|{:>10}|".format('butter','bread','ham')

#%%
#Old style formatting
#-----------------------
#We can even format strings like the old sprintf() style used in C programming language.
# We use the % operator to accomplish this.

x = 12.3456789
print('The value of x is %3.2f' %x)
print('The value of x is %3.4f' %x)

#%%
#%%
#Common Python String Methods
#----------------------------------
#Some of the commonly used methods are len(),strip(),lower(),upper(),split(),join(),find(),replace() etc.

"PrOgRaMiZ".lower()
#%%
"PrOgRaMiZ".upper()
#%%
"This will split all words into a list".split()
#%%
' '.join(['This', 'will', 'join', 'all', 'words', 'into', 'a', 'string'])
#%%
'Happy New Year'.find('ew')
#%%
'Happy New Year'.replace('Happy','Brilliant')

#%%

# Python string methods:  https://www.programiz.com/python-programming/methods/string

# Python String format() : https://www.programiz.com/python-programming/methods/string/format






















