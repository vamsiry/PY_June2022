# 1st topic
#===============
#Interesting facts about strings in Python | Set 1
#1. Strings are Immutable

a = 'Geeks'
print (a)


a[2] = 'E'  # causes error
print (a)    


#%%
#cocatenating strings
#-------------------
a = 'Geeks'
print(a)

a = a + 'for'
print (a) # works fine
#%%
#2. Three ways to create strings:
#------------------------------
#The single quotes and double quotes works same for the string creation.
#Example of single quote and double quote string. Now talking about triple quotes,
#these are used when we have to write a string in multiple lines and printing as
#it is without using any escape sequence.

a = 'Geeks' # string with single quotes
b = "for"   # string with double quotes
c = '''Geeks
a portal
for
geeks'''    # string with triple quotes
print (a)
print (b)
print (c)
 
# Concatenation of strings created using
# different quotes
print (a + b + c )
#%%
#How to print single quote or double quote on screen?
#---------------------------------------------------------
#First one is to use escape character to display the additional quote.
#The second way is by using mix quote, i.e., when we want to print single quote then using double quotes as delimiters and vice-versa.

print ("Hi Mr Geek.")
 
# use of escape sequence
print ("He said, \"Welcome to GeeksforGeeks\"")   
 
print ('Hey so happy to be here')
 
# use of mix quotes
print ('Getting Geeky, "Loving it"')

#%%   set 2
#=======================
#Interesting facts about strings in Python | Set 2 (Slicing)
#------------------------------------------------------------

#The positive index_position denotes the element from the starting(0) and the 
#negative index shows the index from the end(-1).

# A python program to illustrate slicing in strings
 
x = "Geeks at work"
 
# Prints 3rd character beginning from 0
print (x[2])
 
# Prints 7th character
print (x[6])
 
# Prints 3rd character from rear beginning from -1
print (x[-3])
 
# Length of string is 10 so it is out of bound
print (x[15])

#%%
#Slicing
#---------
#Note: We can also slice the string using beginning and only and steps are optional.

# A python program to illustrate
# print substrings of a string
x = "Welcome to GeeksforGeeks"
 
# Prints substring from 2nd to 5th character
print (x[2:5])   
 
# Prints substring stepping up 2nd character 
# from 4th to 10th character
print (x[4:10:2])
 
# Prints 3rd character from rear from 3 to 5
print (x[-5:-3])    

#%% String Methods – Set 1 , Set 2 , Set 3
#=========================================
#set 1
#---------

#Python String Methods | Set 1 (find, rfind, startwith, endwith, islower, 
#isupper, lower, upper, swapcase & title)
#------------------------------------------------------------------------

#1. find(“string”, beg, end) :- This function is used to find the position of the 
#   substring within a string.It takes 3 arguments, substring , starting index( 
#   by default 0) and ending index( by default string length).

#It returns “-1 ” if string is not found in given range.
#It returns first occurrence of string if found.

#2. rfind(“string”, beg, end) :- This function has the similar working as find(),
#   but it returns the position of the last occurrence of string.

str = "geeksforgeeks is for geeks"
str2 = "geeks"
 
# using find() to find first occurrence of str2
# returns 8
print ("The first occurrence of str2 is at : ", end="")
print (str.find(str2, 4))
 

# using rfind() to find last occurrence of str2
# returns 21
print ("The last occurrence of str2 is at : ", end="")
print (str.rfind(str2, 4))

#%%

#3. startswith(“string”, beg, end) :- The purpose of this function is to return true if the function begins with mentioned string(prefix) else return false.

#4. endswith(“string”, beg, end) :- The purpose of this function is to return true if the function ends with mentioned string(suffix) else return false.

str = "geeksforgeeks"
str1 = "geeks"
 
# using startswith() to find if str starts with str1
if    str.startswith(str1):
        print ("str begins with str1")
else :  print ("str does not begin with str1")
 
# using endswith() to find if str ends with str1
if str.endswith(str1):
       print ("str ends with str1")
else : print ("str does not end with str1")

#%%

#5. islower(“string”) :- This function returns true if all the letters in the string are lower cased, otherwise false.

#6. isupper(“string”) :- This function returns true if all the letters in the string are upper cased, otherwise false.

str = "GeeksforGeeks"
str1 = "geeks"
 
# checking if all characters in str are upper cased
if str.isupper() :
       print ("All characters in str are upper cased")
else : print ("All characters in str are not upper cased")
 
# checking if all characters in str1 are lower cased
if str1.islower() :
       print ("All characters in str1 are lower cased")
else : print ("All characters in str1 are not lower cased")

#%%

#7. lower() :- This function returns the new string with all the letters converted into its lower case.

#8. upper() :- This function returns the new string with all the letters converted into its upper case.

#9. swapcase() :- This function is used to swap the cases of string i.e upper case is 
#                 converted to lower case and vice versa.

#10. title() :- This function converts the string to its title case i.e the first letter
#               of every word of string is upper cased and else all are lower cased.


str = "GeeksForGeeks is fOr GeeKs"
 
# Coverting string into its lower case
str1 = str.lower();
print (" The lower case converted string is : " + str1)
 
# Coverting string into its upper case
str2 = str.upper();
print (" The upper case converted string is : " + str2)
 
# Coverting string into its swapped case
str3 = str.swapcase();
print (" The swap case converted string is : " + str3)
 
# Coverting string into its title case
str4 = str.title();
print (" The title case converted string is : " + str4)

#%%
#set 2
#------

#Python String Methods | Set 2 (len, count, center, ljust, rjust, isalpha, isalnum, 
#isspace & join)

#1. len() :- This function returns the length of the string.

#2. count(“string”, beg, end) :- This function counts the occurrence of mentioned substring
#   in whole string. This function takes 3 arguments, substring, beginning position( 
#   by default 0) and end position(by default string length).

str = "geeksforgeeks is for geeks"
  
# Printing length of string using len()
print (" The length of string is : ", len(str));
 
# Printing occurrence of "geeks" in string
# Prints 2 as it only checks till 15th element
print (" Number of appearance of ""geeks"" is : ",end="")
print (str.count("geeks",0,15))

#%%

#3. center() :- This function is used to surround the string with a character repeated 
#   both sides of string multiple times. By default the character is a space. 
#   Takes 2 arguments, length of string and the character.

#4. ljust() :- This function returns the original string shifted to left that has a
#   character at its right. It left adjusts the string. By default the character is
#   space. It also takes two arguments, length of string and the character.

#5. rjust() :- This function returns the original string shifted to right that has 
#   a character at its left. It right adjusts the string. By default the character 
#   is space. It also takes two arguments, length of string and the character.


str = "geeksforgeeks"
  
# Printing the string after centering with '-'
print ("The string after centering with '-' is : ",end="")
print ( str.center(20,'-'))
 
# Printing the string after ljust()
print ("The string after ljust is : ",end="")
print ( str.ljust(20,'-'))
 
# Printing the string after rjust()
print ("The string after rjust is : ",end="")
print ( str.rjust(20,'-'))

#%%

#6. isalpha() :- This function returns true when all the characters in the string 
#   are alphabets else returns false.

#7. isalnum() :- This function returns true when all the characters in the string 
#   are combination of numbers and/or alphabets else returns false.

#8. isspace() :- This function returns true when all the characters in the string
#   are spaces else returns false.

str = "geeksforgeeks"
str1 = "123"
  
# Checking if str has all alphabets 
if (str.isalpha()):
       print ("All characters are alphabets in str")
else : print ("All characters are not alphabets in str")
 
# Checking if str1 has all numbers
if (str1.isalnum()):
       print ("All characters are numbers in str1")
else : print ("All characters are not numbers in str1")
 
# Checking if str1 has all spaces
if (str1.isspace()):
       print ("All characters are spaces in str1")
else : print ("All characters are not spaces in str1")

#%%

#9. join() :- This function is used to join a sequence of strings mentioned in its arguments with the string.

str = "_"
str1 = ( "geeks", "for", "geeks" )
 
# using join() to join sequence str1 with str
print ("The string after joining is : ", end="")
print ( str.join(str1))

#%% set 3

#Python String Methods | Set 3 (strip, lstrip, rstrip, min, max, maketrans,
# translate & relplace)

#1. strip() :- This method is used to delete all the leading and trailing characters mentioned in its argument.

#2. lstrip() :- This method is used to delete all the leading characters mentioned in its argument.

#3. rstrip() :- This method is used to delete all the trailing characters mentioned in its argument.


str = "-----geeksforgeeks---"
 
# using strip() to delete all '-'
print ( " String after stripping all '-' is : ", end="")
print ( str.strip('-') )
 
# using lstrip() to delete all trailing '-'
print ( " String after stripping all leading '-' is : ", end="")
print ( str.lstrip('-') )
 
# using rstrip() to delete all leading '-'
print ( " String after stripping all trailing '-' is : ", end="")
print ( str.rstrip('-') )

#%%

#4. min(“string”) :- This function returns the minimum value alphabet from string.

#5. max(“string”) :- This function returns the maximum value alphabet from string.


str = "geeksforgeeks"
 
# using min() to print the smallest character
# prints 'e'
print ("The minimum value character is : " + min(str));
 
# using max() to print the largest character
# prints 's'
print ("The maximum value character is : " + max(str));

#%%

#6. maketrans() :- It is used to map the contents of string 1 with string 2 with 
#   respective indices to be translated later using translate().

#7. translate() :- This is used to swap the string elements mapped with the help of maketrans().

#from string import maketrans # for maketrans() not avaliablein python 3.6
 
str = "geeksforgeeks"   
 
str1 = "gfo"
str2 = "abc"
 
# using maktrans() to map elements of str2 with str1
mapped = "".maketrans( str1, str2 );
 
# using translate() to translate using the mapping
print ("The string after translation using mapped elements is : ")
print  (str.translate(mapped)) ;

print("Swap vowels for numbers.".translate(str.maketrans('aeiou', '12345')))


#ex : 2
intab = 'aeiou'
outtab = '12345'

s = 'this is string example....wow!!!'

print(s.translate({ord(x): y for (x, y) in zip(intab, outtab)}))

#%%

#8. replace() :- This function is used to replace the substring with a new substring
#   in the string. This function has 3 arguments. The string to replace, new string 
#   which would replace and max value denoting the limit to replace action
# ( by default unlimited ).
 
str = "nerdsfornerds is for nerds"
 
str1 = "nerds"
str2 = "geeks"
 
# using replace() to replace str2 with str1 in str
# only changes 2 occurrences 
print ("The string after replacing strings is : ", end="")
print (str.replace( str1, str2, 2)) 

#%% topic 3

#Logical Operators on String in Python
#======================================

str1 = ''
str2 = 'geeks'
 
# repr is used to print the string along with the quotes
print (repr(str1 and str2)) # Returns str1 
print (repr(str2 and str1)) # Returns str1
print (repr(str1 or str2)) # Returns str2 
print (repr(str2 or str1)) # Returns str2
 
str1 = 'for'
 
print (repr(str1 and str2)) # Returns str2 
print (repr(str2 and str1)) # Returns str1
print (repr(str1 or str2)) # Returns str1 
print (repr(str2 or str1)) # Returns str2

# The output of the boolean operations between the strings depends on following things:
# Python considers empty strings as having boolean value of ‘false’ and non-empty string as having boolean value of ‘true’.
# For ‘and’ operator if left value is true, then right value is checked and returned. If left value is false, then it is returned
# For ‘or’ operator if left value is true, then it is returned, otherwise if left value is false, then right value is returned.

#Note that the bitwise operators (| , &) don’t work for strings.
 
#%%

#How to split a string in Python?

#Splitting a string by some delimiter is a very common task. For example, we have a
# comma separated list of items from a file and we want individual items in an array. 

 
 # // regexp is the delimiting regular expression; 
 # // limit is limit the number of splits to be made 
 # str.split(regexp = "", limit = string.count(str)) 

line = "Geek1 \nGeek2 \nGeek3";

#line = "Geek1 Geek2 Geek3";

print (line.split())
print (line.split(' ', 1))

#%%
#String Formatting in Python using %(formatting operator)
#1) Using %
#2) Using {}
#3) Using Template Strings

#The formatting using % is similar to that of ‘printf’ in C programming language.
#%d – integer
#%f – float
#%s – string
#%x – hexadecimal
#%o – octal

# Initialize variable as a string
variable = '15'
string = "Variable as string = %s" %(variable)
print (string)
  
# Printing as raw data
print ("Variable as raw data = %r" %(variable))
  
# Convert the variable to integer
# And perform check other formatting options 
variable = int(variable) # Without this the below statement will give error.                       
string = "Variable as integer = %d" %(variable)
print (string)
print ("Variable as float = %f" %(variable))
 
# printing as any string or char after a mark
# here i use mayank as a string
print ("Variable as printing with special char = %cmayank" %(variable))
 
print ("Variable as hexadecimal = %x" %(variable))
print ("Variable as octal = %o" %(variable))

#%%

#String Template Class in Python

#In String module, Template Class allows us to create simplified syntax for output 
#specification. The format uses placeholder names formed by $ with valid Python 
#identifiers (alphanumeric characters and underscores). Surrounding the placeholder
#with braces allows it to be followed by more alphanumeric letters with no intervening
#spaces. Writing $$ creates a single escaped $:

# A Simple Python templaye example
from string import Template
 
# Create a template that has placeholder for value of x
t = Template("x is $x")
 
# Substitute value of x in above template
print (t.substitute({"x" : 1}))

#%%

#Following is another example where we import names and marks of students from
# a list and print them using template.

from string import Template
 
# List Student stores the name and marks of three students
Student = [('Ram',90), ('Ankit',78), ('Bob',92)]

# We are creating a basic structure to print the name and
# marks of the students.
t = Template('Hi $name, you have got $marks marks')
 
for i in Student:
     print (t.substitute(name = i[0], marks = i[1]))


#The substitute() method raises a KeyError when a placeholder is not supplied in
# a dictionary or a keyword argument. For mail-merge style applications, user 
#supplied data may be incomplete and the safe_substitute() method may be more
# appropriate — it will leave placeholders unchanged if data is missing:

#Another application for template is separating program logic from the details of
# multiple output formats. This makes it possible to substitute custom templates 
#for XML files, plain text reports, and HTML web reports.

#Note that there are other ways also to print formatted output like %d for integer,
# %f for float, etc 
     
#%%
























































