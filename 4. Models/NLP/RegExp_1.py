# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:06:41 2020

@author: rvamsikrishna
"""

#FlashText – A library faster than Regular Expressions for NLP tasks

#https://www.analyticsvidhya.com/blog/2017/11/flashtext-a-library-faster-than-regular-expressions/
#%%
#https://www.analyticsvidhya.com/blog/2015/06/regular-expression-python/

#What is Regular Expression and how is it used?
#------------------------------------------------

#regular expression is a sequence of character(s) mainly used to find 
#and replace patterns in a string or file
 
#Regular expressions use two types of characters:
#--------------------------------------------------------
#a) Meta characters: As the name suggests, these characters have a 
#special meaning, similar to * in wild card.
 
#b) Literals (like a,b,1,2…)


#The most common uses of regular expressions are:
#--------------------------------------------------
#Search a string (search and match)
#Finding a string (findall)
#Break string into a sub strings (split)
#Replace part of a string (sub)

#What are various methods of Regular Expressions?
#---------------------------------------------------

# ‘re’ package provides multiple methods,Here are the most commonly used methods.
#re.match()
#re.search()
#re.findall()
#re.split()
#re.sub()
#re.compile()

#%%
#%%
#%%
#re.match(pattern, string): 
#==========================
#This method finds match if it occurs at start of the string

# Importing libraries
import re

# matching AV in the given sentence
result = re.match(r'AV', 'AV Analytics Vidhya AV')
print ('\n',result)

#%%
# The output shows that pattern match has been found. To print the matching 
#string we’ll use method group (It helps to return the matching string). 
#Use “r” at the start of the pattern string, it designates a python raw string.

# printing the matching string
result = re.match(r'AV', 'AV Analytics Vidhya AV')
print ('\nMatching string :',result.group(0))

#%%
# Let’s now find the word ‘Analytics’ in the given string. Here we see that string is
# not starting with ‘AV’ so it should return no match. Let’s see what we get:

# matching Analytics in the given sentence
result = re.match(r'Analytics', 'AV Analytics Vidhya AV')
print ('\nResult :', result)

#%%
# There are methods like start() and end() to know the start and end position
# of matching pattern in the string.

result = re.match(r'AV', 'AV Analytics Vidhya AV')
print ('\nStarting position of the match :',result.start())
print ('Ending position of the match :',result.end())

#%%
#re.search(pattern, string):
#================================

#It is similar to match() but it doesn’t restrict us to find 
#matches at the beginning of the string only

result = re.search(r'Analytics', 'AV Analytics Vidhya AV')
print(result.group(0))

#%%
#re.findall (pattern, string):
#===============================

#It helps to get a list of all matching patterns.
# It has no constraints of searching from start or end. 
#If we will use method findall to search
# ‘AV’ in given string it will return both occurrence of AV. While
# searching a string, I would recommend you to use re.findall() always, 
#it can work like re.search() and re.match() both.

result = re.findall(r'AV', 'AV Analytics Vidhya AV')
print (result)

#%%
#re.split(pattern, string, [maxsplit=0]):
#=============================================

#This methods helps to split string by the occurrences of given pattern.

result=re.split(r'y','Analytics')
result

#%%
result=re.split(r'i','Analytics Vidhya')
print (result)

#%%
result=re.split(r'i','Analytics Vidhya',maxsplit=1)
result

#%%
#re.sub(pattern, repl, string):
#===================================

#It helps to search a pattern and replace with a new sub string. 
#If the pattern is not found, string is returned unchanged.

result=re.sub(r'India','the World','AV is largest Analytics community of India')
result

#%%
##re.compile(pattern, repl, string):
#======================================

#We can combine a regular expression pattern into pattern objects, 
#which can be used for pattern matching. It also helps to search a
# pattern again without rewriting it.

import re
pattern=re.compile('AV')

result=pattern.findall('AV Analytics Vidhya AV')
print (result)

result2=pattern.findall('AV is largest analytics community of India')
print (result2)

#%%

#Till now,  we looked at various methods of regular expression using a constant 
#pattern (fixed characters).
# But, what if we do not have a constant search pattern and we want to
# return specific set of characters (defined by a rule) from a string?

#This can be solved by defining an expression with the help of
# pattern operators (meta  and literal characters).

#most commonly used operators?
#================================

# =============================================================================
# Operators 	Description
# . 	 Matches with any single character except newline ‘\n’.
# ? 	 match 0 or 1 occurrence of the pattern to its left
# + 	 1 or more occurrences of the pattern to its left
# * 	 0 or more occurrences of the pattern to its left
# \w 	 Matches with a alphanumeric character whereas \W (upper case W) matches non alphanumeric character.
# \d 	  Matches with digits [0-9] and /D (upper case D) matches with non-digits.
# \s 	 Matches with a single white space character (space, newline, return, tab, form) and \S (upper case S) matches any non-white space character.
# \b 	 boundary between word and non-word and /B is opposite of /b
# [..] 	 Matches any single character in a square bracket and [^..] matches any single character not in square bracket
# \ 	 It is used for special meaning characters like \. to match a period or \+ for plus sign.
# ^ and $ 	 ^ and $ match the start or end of the string respectively
# {n,m} 	 Matches at least n and at most m occurrences of preceding expression if we write it as {,m} then it will return at least any minimum occurrence to max m preceding expression.
# a| b 	 Matches either a or b
# ( ) 	Groups regular expressions and returns matched text
# \t, \n, \r 	 Matches tab, newline, return
 


#%%
#%%
#%%

#Some Examples of Regular Expressions
#========================================

#Problem 1: Return the first word of a given string
#======================================================

#Solution-1  Extract each character (using “\w“)
import re
result=re.findall(r'.','AV is largest Analytics community of India')
print(result)

#%%
#Above, space is also extracted, now to avoid it use “\w” instead of “.“.
result=re.findall(r'\w','AV is largest Analytics community of India')
print (result)

#%%
#Solution-2  Extract each word (using “*” or “+“)

result=re.findall(r'\w*','AV is largest Analytics community of India')
print (result)

#%%
#Again, it is returning space as a word because “*” returns zero 
#or more matches of pattern to its left. Now to remove spaces 
#we will go with “+“.
result=re.findall(r'\w+','AV is largest Analytics community of India')
print (result)

#%%
#Solution-3 Extract each word (using “^“)

#If we will use “^”, it will return the word from the start of the string.

result=re.findall(r'^\w+','AV is largest Analytics community of India')
print (result)

#%%
#If we will use “$” instead of “^”, it will return the word from 
#the end of the string.

result=re.findall(r'\w+$','AV is largest Analytics community of India')
print (result)


#%%
#%%
#Problem 2: Return the first two character of each word
#=========================================================

#Solution-1  Extract consecutive two characters of each word,
# excluding spaces (using “\w“)
result=re.findall(r'\w\w','AV is largest Analytics community of India')
print(result)
    
#%%
#Solution-2  Extract consecutive two characters those 
#available at start of word boundary (using “\b“)
result=re.findall(r'\b\w.','AV is largest Analytics community of India')
print (result)

#%%
#%%
#Problem 3: Return the domain type of given email-ids
#======================================================

#Solution-1  Extract all characters after “@”
result=re.findall(r'@\w+','abc.test@gmail.com, xyz@test.in, test.first@analyticsvidhya.com, first.test@rest.biz') 
print(result)
 
#%%
#Above, you can see that “.com”, “.in” part is not extracted. 
#To add it, we will go with below code.
result=re.findall(r'@\w+.\w+','abc.test@gmail.com, xyz@test.in, test.first@analyticsvidhya.com, first.test@rest.biz')
print(result)

#%%
#Solution – 2 Extract only domain name using “( )”
result=re.findall(r'@\w+.(\w+)','abc.test@gmail.com, xyz@test.in, test.first@analyticsvidhya.com, first.test@rest.biz')
print(result)

#%%
#%%
#Problem 4: Return date from given string
#============================================

#Here we will use “\d” to extract digit.
result=re.findall(r'\d{2}-\d{2}-\d{4}','Amit 34-3456 12-05-2007, XYZ 56-4532 11-11-2011, ABC 67-8945 12-01-2009')
print (result)

#%%

#If you want to extract only year again parenthesis “( )” will help you.

result=re.findall(r'\d{2}-\d{2}-(\d{4})','Amit 34-3456 12-05-2007, XYZ 56-4532 11-11-2011, ABC 67-8945 12-01-2009')
print(result)

#%%
#%%
#Problem 5: Return all words of a string those starts with vowel
#==============================================================

#Solution-1  Return each words
result=re.findall(r'\w+','AV is largest Analytics community of India')
print(result)

#%%
#Solution-2  Return words starts with vowel alphabets (using [])
result=re.findall(r'[aeiouAEIOU]\w+','AV is largest Analytics community of India')
print (result)

#%%
#Above you can see that it has returned “argest” and “ommunity” from 
#the mid of words. To drop these two, we need to use “\b” for word
# boundary.
result=re.findall(r'\b[aeiouAEIOU]\w+','AV is largest Analytics community of India')
print(result) 

#%%
#In similar ways, we can extract words those starts with 
#consonant using “^” within square bracket.
result=re.findall(r'\b[^aeiouAEIOU]\w+','AV is largest Analytics community of India')
print(result)

#%%
#Above you can see that it has returned words starting with space.
#To drop it from output, include space in square bracket[].
result=re.findall(r'\b[^aeiouAEIOU ]\w+','AV is largest Analytics community of India')
print(result)

#%%
#%%
#Problem 6: Validate a phone number (phone number
# must be of 10 digits and starts with 8 or 9) 
#================================================
import re
li=['9999999999','999999-999','99999x9999']
for val in li:
 if re.match(r'[8-9]{1}[0-9]{9}',val) and len(val) == 10:
     print('yes')
 else:
     print('no')


#%%
#%%
#Problem 7: Split a string with multiple delimiters
#==================================================
import re
line = 'asdf fjdk;afed,fjek,asdf,foo' # String has multiple delimiters (";",","," ").
result= re.split(r'[;,\s]', line)
print(result)     

#%%
#We can also use method re.sub() to replace these 
#multiple delimiters with one as space ” “.

import re
line = 'asdf fjdk;afed,fjek,asdf,foo'
result= re.sub(r'[;,\s]',' ', line)
print(result)

#%%
#%%
#Problem 8: Retrieve Information from HTML file
#================================================

# Here we need to extract information available between 
#<td> and </td> except the first numerical index

#I have assumed here that below html code is stored in a string str.

str = """<tr align="center"><td>1</td> <td>Noah</td> <td>Emma</td></tr>
<tr align="center"><td>2</td> <td>Liam</td> <td>Olivia</td></tr>
<tr align="center"><td>3</td> <td>Mason</td> <td>Sophia</td></tr>
<tr align="center"><td>4</td> <td>Jacob</td> <td>Isabella</td></tr"""

result=re.findall(r'<td>\w+</td>\s<td>(\w+)</td>\s<td>(\w+)</td>',str)
print(result)

#%%
#You can read html file using library urllib2 (see below code).
import urllib2
response = urllib2.urlopen('')
html = response.read()

#%%
































  