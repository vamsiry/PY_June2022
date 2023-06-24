# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 06:07:43 2022

@author: rvamsikrishna
"""

#%%
#Python String capitalize()
#-------------------------------
#capitalize() method converts the first character of a string to an 
#uppercase letter and all other alphabets to lowercase.

sentence = "uppercase letter and all other alphabets to lowercase."

# converts first character to uppercase and others to lowercase
sentence.capitalize()

#%%
#Python String center()
#-----------------------
#The center() method returns a new centered string after padding it 
#with the specified character.

sentence = "Python is awesome"

# returns the centered padded string of length 50 
print(sentence.center(50, '*'))

print(sentence.center(50))

#%%
#Python String casefold()
#---------------------------
#casefold() method converts all characters of the string into 
#lowercase letters and returns a new string. casefold() method
# converts all characters of the string into lowercase letters 
#and returns a new string. 

sentence = "PYTHON is AWESOME"
sentence.casefold()
sentence.lower()

#note : The casefold() method is similar to the lower() method but it is 
#more aggressive. This means the casefold() method converts more 
#characters into lower case compared to lower() .

#For example, the German letter ß is already lowercase so, the lower() 
#method doesn't make the conversion.

#But the casefold() method will convert ß to its equivalent character ss.

#%%
#%%
#Python String count()
#---------------------
#The count() method returns the number of occurrences of a substring 
#in the given string.

message = 'python is popular programming language'

# number of occurrence of 'p'
print('Number of occurrence of p:', message.count('p'))

#string.count(substring, start=..., end=...)
#count() method only requires a single parameter for execution. However,
# it also has two optional parameters:
#substring - string whose count is to be found.
#start (Optional) - starting index within the string where search starts.
#end (Optional) - ending index within the string where search ends.

#%%
#count() method returns the number of occurrences of the substring
# in the given string.

string = "Python is awesome, isn't it?"
substring = "is"

string.count(substring)

#%%
#Example 2: Count number of occurrences of a given substring using start and end
 
string = "Python is awesome, isn't it?"
substring = "i"

# count after first 'i' and before the last 'i'
string.count(substring, 8, 25)
        
#%%
#%%
#Python String startswith() and endswith()
#---------------------------------------------
# endswith() method returns True if a string ends with the specified suffix. 
#If not, it returns False.

message = 'Python is fun'

# check if the message ends with fun
print(message.endswith('fun'))

print(message.startswith('P'))

#%%
#str.endswith(suffix[, start[, end]])
#suffix - String or tuple of suffixes to be checked
#start (optional) - Beginning position where suffix is to be checked within the string.
#end (optional) - Ending position where suffix is to be checked within the string.
text = "Python programming is easy to learn."
print(text.endswith('learn.', 7))
print(text.endswith('is', 7, 26))
print(text.endswith('easy', 7, 26))

#%%
#Passing Tuple to endswith()
#It's possible to pass a tuple suffix to the endswith() method in Python.
#If the string ends with any item of the tuple, endswith() returns True. 
#If not, it returns False

print(text.endswith(('programming', 'python')))

print(text.endswith(('python', 'easy', 'java')))

print(text.endswith(('is', 'an'), 0, 14))


#%%
#%%
#Python String expandtabs()
#-------------------------------
#The expandtabs() returns a string where all '\t' characters are replaced 
#with whitespace characters until the next multiple of tabsize parameter.

#The expandtabs() takes an integer tabsize argument. The default tabsize is 8.

str = 'xyz\t12345\tabc'

print(str.expandtabs())
print(str.expandtabs(2))
print(str.expandtabs(4))
print(str.expandtabs(6))

#%%
#%%
#Python String encode()
#--------------------
#encode() method returns an encoded version of the given string.

#It returns an utf-8 encoded version of the string. 
#In case of failure, it raises a UnicodeDecodeError exception.

title = 'Python Programming'

# change encoding to utf-8
print(title.encode())

#string.encode(encoding='UTF-8',errors='strict')
#encoding - the encoding type a string has to be encoded to
#errors--response when encoding fails. 
#https://www.programiz.com/python-programming/methods/string/encode

print(string.encode("ascii", "ignore"))

print(string.encode("ascii", "replace"))

#String Encoding
#Since Python 3.0, strings are stored as Unicode, i.e. each character 
#in the string is represented by a code point. So, each string is just a 
#sequence of Unicode code points.

#For efficient storage of these strings, the sequence of code points is 
#converted into a set of bytes. The process is known as encoding.


#%%
#Python String find()
#-----------------------
#find() method returns the index of first occurrence of the substring 
#(if found). If not found, it returns -1.

message = 'Python is a fun programming language'
message.find('fun')

#%%
#str.find(sub[, start[, end]] )
#sub - It is the substring to be searched in the str string.
#start and end (optional) - The range str[start:end] within which 
#substring is searched.

quote = 'Let it be, let it be, let it be'
print(quote.find('let it'))
print(quote.find('small')) ## find returns -1 if substring not found

#%%
# How to use find()
if (quote.find('be,') != -1):

    print("Contains substring 'be,'")
else:
    print("Doesn't contain substring")

#%%
quote = 'Do small things with great love'

# Substring is searched in 'hings with great love'
print(quote.find('small things', 10))


# Substring is searched in ' small things with great love' 
print(quote.find('small things', 2))

# Substring is searched in 'hings with great lov'
print(quote.find('o small ', 10, -1))


# Substring is searched in 'll things with'
print(quote.find('things ', 6, 20))

#%%
#%%
#%%
#Python String index()
#------------------------
#index() method returns the index of a substring inside the string (if found).
# If the substring is not found, it raises an exception.
text = 'Python is fun'
text.index('is')

#str.index(sub[, start[, end]] )
#If substring doesn't exist inside the string, it raises a ValueError exception.

#The index() method is similar to the find() method for strings.
#The only difference is that find() method returns -1 if the substring is
# not found, whereas index() throws an exception.

sentence = 'Python programming is fun.'
print(sentence.index('is fun'))
print(sentence.index('Java'))
print(sentence.index('ing', 10))
print(sentence.index('g is', 10, -4))
print(sentence.index('fun', 7, 18))
    
#%%
#%%
#Python String format()
#--------------------------
#format() method formats the given string into a nicer output in Python.

#template.format(p0, p1, ..., k0=v0, k1=v1, ...)

#Here, p0, p1,... are positional arguments and, 
#k0, k1,... are keyword arguments with values v0, v1,... respectively.
    
#template is a mixture of format codes with placeholders for the arguments.


#String format() Parameters

#format() method takes any number of parameters. But, is divided 
#into two types of parameters:

#Positional parameters - list of parameters that can be accessed with 
#index of parameter inside curly braces {index}
#
#Keyword parameters - list of parameters of type key=value, that can
#be accessed with key of parameter inside curly braces {key}

#Basic formatting with format()
#Example 1: Basic formatting for default, positional and keyword arguments

# default arguments
print("Hello {}, your balance is {}.".format("Adam", 230.2346))

# positional arguments
print("Hello {0}, your balance is {1}.".format("Adam", 230.2346))

# keyword arguments
print("Hello {name}, your balance is {blc}.".format(name="Adam", blc=230.2346))

# mixed arguments
print("Hello {0}, your balance is {blc}.".format("Adam", blc=230.2346))

#%%
#Example 2: Simple number formatting

# integer arguments
print("The number is:{:d}".format(123))

# float arguments
print("The float number is:{:f}".format(123.4567898))

# octal, binary and hexadecimal format
print("bin: {0:b}, oct: {0:o}, hex: {0:x}".format(12))

#%%
#Example 3: Number formatting with padding for int and floats

# integer numbers with minimum width
print("{:5d}".format(12))

# width doesn't work for numbers longer than padding
print("{:2d}".format(1234))

# padding for float numbers
print("{:8.3f}".format(12.2346))

# integer numbers with minimum width filled with zeros
print("{:05d}".format(12))

# padding for float numbers filled with zeros
print("{:08.3f}".format(12.2346))

#%%
#Example 4: Number formatting for signed numbers

# show the + sign
print("{:+f} {:+f}".format(12.23, -12.23))

# show the - sign only
print("{:-f} {:-f}".format(12.23, -12.23))

# show space for + sign
print("{: f} {: f}".format(12.23, -12.23))

#%%
#Example 5: Number formatting with left, right and center alignment

# integer left alignment padding with zeros 
print("{:<11d}".format(12))

# integer left alignment padding with zeros 
print("{:<011d}".format(12))

# float numbers with center alignment
print("{:^11.3f}".format(12.2346))

# float numbers with center alignment
print("{:=8.3f}".format(-12.2346))

# integer numbers with right alignment
print("{:>11d}".format(12))

# integer numbers with right alignment padding with zeros
print("{:>011d}".format(12))


#%%
#Example 6: String formatting with padding and alignment

# string padding with left alignment
print("{:11}".format("cat"))

# string padding with right alignment
print("{:>11}".format("cat"))

# string padding with right alignment
print("{:*>11}".format("cat"))

# string padding with center alignment
print("{:^11}".format("cat"))

# string padding with center alignment and '*' padding character
print("{:*^11}".format("cat"))

#%%
#Example 7: Truncating strings with format()

# truncating strings to 3 letters
print("{:.3}".format("caterpillar"))

# truncating strings to 3 letters and padding
print("{:*>5.3}".format("caterpillar"))

# truncating strings to 3 letters,padding and center alignment
print("{:^5.3}".format("caterpillar"))

#%%
#Formatting class and dictionary members using format()

#Python internally uses getattr() for class members in the form ".age". 
#And, it uses __getitem__() lookup for dictionary members in the form "[index]".

#Example 8: Formatting class members using format()
# define Person class
class Person:
    age = 23
    name = "Adam"

# format age
print("{p.name}'s age is: {p.age}".format(p=Person()))

#%%
#Example 9: Formatting dictionary members using format()
# define Person dictionary
person = {'age': 23, 'name': 'Adam'}

# format age
print("{p[name]}'s age is: {p[age]}".format(p=person))

#or

# format age
print("{name}'s age is: {age}".format(**person))

#%%
#Arguments as format codes using format()
#You can also pass format codes like precision, alignment, fill character 
#as positional or keyword arguments dynamically.

#Example 10: Dynamic formatting using format()
# dynamic string format template
string = "{:{fill}{align}{width}}"

# passing format codes as arguments
print(string.format('cat', fill='*', align='^', width=5))

# dynamic float format template
num = "{:{align}{width}.{precision}f}"

# passing format codes as arguments
print(num.format(123.236, align='<', width=8, precision=2))

#%%
#Extra formatting options with format()
#Example 11: Type-specific formatting with format() and overriding 
#__format__() method

import datetime
# datetime formatting
date = datetime.datetime.now()
print("It's now: {:%Y/%m/%d %H:%M:%S}".format(date))

# complex number formatting
complexNumber = 1+2j
print("Real part: {0.real} and Imaginary part: {0.imag}".format(complexNumber))

# custom __format__() method
class Person:
    def __format__(self, format):
        if(format == 'age'):
            return '23'
        return 'None'

print("Adam's age is: {:age}".format(Person()))

#%%
#Example 12: __str()__ and __repr()__ shorthand !r and !s using format()
# __str__() and __repr__() shorthand !r and !s
print("Quotes: {0!r}, Without Quotes: {0!s}".format("cat"))

# __str__() and __repr__() implementation for class
class Person:
    def __str__(self):
        return "STR"
    def __repr__(self):
        return "REPR"

print("repr: {p!r}, str: {p!s}".format(p=Person()))

#%%
#%%
#Python String isalnum()
#--------------------------
#The isalnum() method returns True if all characters in the string are 
#alphanumeric (either alphabets or numbers). If not, it returns False.

name1 = "Python3"
print(name1.isalnum()) #True

name2 = "Python 3"
print(name2.isalnum()) #False

#%%
#SYNTEX -- string.isalnum()
#--------------------------------
# contains either numeric or alphabet
string1 = "M234onica"
print(string1.isalnum()) # True 

# contains whitespace
string2 = "M3onica Gell22er"
print(string2.isalnum()) # False

# contains non-alphanumeric character 
string3 = "@Monica!"
print(string3.isalnum()) # False 

#%%
#%%
#Python String isalpha()
#-------------------------
#isalpha() method returns True if all characters in the 
#string are alphabets. If not, it returns False.

#True if all characters in the string are alphabets (can be both
# lowercase and uppercase).

#False if at least one character is not alphabet.

name = "Monica"
print(name.isalpha())

# contains whitespace
name = "Monica Geller"
print(name.isalpha())

# contains number
name = "Mo3nicaGell22er"
print(name.isalpha())

#%%
#%%
#Python String isdecimal()
#-----------------------------
#isdecimal() method returns True if all characters in a string are
# decimal characters. If not, it returns False.

#True if all characters in the string are decimal characters.
#False if at least one character is not decimal character.

s = "28212"
print(s.isdecimal())

# contains alphabets
s = "32ladk3"
print(s.isdecimal())

# contains alphabets and spaces
s = "Mo3 nicaG el l22er"
print(s.isdecimal())

#%%
#%%
#Python String isdigit()
#---------------------------
#isdigit() method returns True if all characters in a string are digits.
# If not, it returns False.

#True if all characters in the string are digits.
#False if at least one character is not a digit.

str1 = '342'
print(str1.isdigit())

str2 = 'python'
print(str2.isdigit())

#%%
#%%
#Python String isidentifier()
#--------------------------------
# isidentifier() method returns True if the string is a valid 
#identifier in Python. If not, it returns False.

#True if the string is a valid identifier
#False if the string is not a invalid identifier

str = 'Python'
print(str.isidentifier())

str = 'Py thon'
print(str.isidentifier())

str = '22Python'
print(str.isidentifier())

str = ''
print(str.isidentifier())

#%%
#%%
#Python String islower()
#-------------------------
#islower() method returns True if all alphabets in a string are 
#lowercase alphabets. If the string contains at least one 
#uppercase alphabet, it returns False.

#True if all alphabets that exist in the string are lowercase alphabets.
#False if the string contains at least one uppercase alphabet.
s = 'this is good'
print(s.islower())

s = 'th!s is a1so g00d'
print(s.islower())

s = 'this is Not good'
print(s.islower())

#%%
#%%


























