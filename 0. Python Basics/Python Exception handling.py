# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 18:35:46 2022

@author: rvamsikrishna
"""


#%%
#File Avaliability checking 
def FileCheck(fn):
    try:
      open(fn, "r")
      return 1
    except IOError:
        print ("Error: File does not appear to exist.")
        return 0

result = FileCheck("testfile")
print(result)

#%%
#%%
#%%

#Files and Exceptions in Python
#----------------------------------
#https://www.section.io/engineering-education/files-and-exceptions-in-python/

#Files are identified locations on a disk where associated data is stored. 
#Working with files will make your programs fast when analyzing masses of data.

#Exceptions are special objects that any programming language
# uses to manage errors that occur when a program is running.


#Reading from a file
#-----------------
#Let’s create a text file containing a list of years from 2020 to 2022 
#using an editor and save it in the same folder that stores our Python files as years.txt.

#The years.txt file should have the following text:

#Below is a program that opens the above file, reads it and prints the data in the file:
with open('years.txt') as file_object:
    contents = file_object.read()
    print(contents)

#Python looks for the years.txt file in the folder, where our Python file is stored.
    
#The open() function returns an object representing the file (years.txt) 
#which is then stored in variable file_object.
    
#The keyword with closes the file when access to it is no longer need.
    
#The read() method is used to read the whole data in the file and store it in contents.

#%%
#Working with the contents of a file
#--------------------------------------
with open('years.txt') as file_object:
    lines = file_object.readlines()
yrs_string = '' # create a variable yrs_string to hold the digits of years
for line in lines: # create a loop that adds each line of digits to yrs_string
     yrs_string += line.rstrip() #.rstrip() removes the newline character from each line
print(yrs_string) # print this string
print(len(yrs_string)) # print how long the string is     

#NOTE: Python treats all text in a text file as a string. If you read a 
#number from a file and you want to carry out arithmetic operations, 
#convert it to float using the float() function or integer using the 
#int() function.

#%%
#Writing to a file
#-----------------------

#When writing text to a file, we use the open() function with two arguments 
#- the first argument is the filename, while the second argument is the 
#mode in which you want to open the file.

#Read mode (‘r’)
#Write mode (‘w’)
#Append mode (‘a’)
#Read and write (‘r+’)

#%%
#Writing to an empty file
#-------------------------
with open('student.txt', 'w') as file_object:
    file_object.write("My name is Felix.")
    
#NOTE: When opening a file in ‘w’ mode and the file exists, Python will delete the file before returning the file object.

#%%
#Appending to a file
#--------------------
#To add data into a file, open it in append mode. Any data you write 
#will be placed at the end of the file.
    
#Let’s add some lines in the student.txt file:
with open('student.txt', 'a') as file_object: #’a’ argument to open the file for appending
      file_object.write("I am 6 years old\n")
      file_object.write("I love playing games\n")
      
#%%
#%%      
#Exceptions
#-----------------
#Exceptions are unique objects that Python uses to control errors 
#that occur when a program is running. Exceptions errors arise 
#when code is correct syntactically but Python programs produce an error.
      
#Python creates an exception object whenever these mistakes occur. 

#When we write code that deals with the exception, our programs will 
#continue running, even if an error is thrown. 

#If we don’t, our programs will stop executing and show a 
#trace-back, which is very hard for a user to understand.

      
#Python uses the try-except-finally block to control exceptions.
#------------------------------------------------------------------      
# A try-except block informs Python how to act when an exception emerges.
# Our programs will continue to execute even if things go wrong.
      
      
 #%%     
#Handling the ZeroDivisionError exception
#--------------------------------------
#Since Python cannot divide a number by zero, it reports an error
# in the trace-back as ZeroDivisionError, which is an exception object.
# This kind of object responds to a scenario where Python can’t do 
#what we asked it to.
      
#If you think an error might occur, use the try-except block to 
#control the exception that may be raised.
      
try:
    print(6/0)
except ZeroDivisionError:
    print("You can’t divide by zero!") # You can’t divide by zero!

#%%
#Handling the FileNotFoundError exception
#-----------------------------------------------
    
filename = 'John.txt'
with open(filename) as f_obj:
    contents = f_obj.read()

#In this example, the open() function creates the error. 
#To solve this error, use the try block just before the line, which 
#involves the open() function:
    
filename = 'John.txt'
try:
    with open(filename) as f_obj:
        contents = f_obj.read()
except FileNotFoundError:
    msg = "Sorry, the file "+ filename + "does not exist."
    print(msg) # Sorry, the file John.txt does not exist.
    

#%%
#%%
#%%
#Catching Exceptions in Python
#---------------------------------    
    
#https://www.programiz.com/python-programming/exception-handling
    
#The critical operation which can raise an exception is placed inside
# the try clause. The code that handles the exceptions is written 
#in the except clause.
    
# import module sys to get the type of exception
import sys

randomList = ['a', 0, 2]

for entry in randomList:
    try:
        print("The entry is", entry)
        r = 1/int(entry)
        break
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")
        print("Next entry.")
        print()
print("The reciprocal of", entry, "is", r)

#%%
#Since every exception in Python inherits from the base Exception class,
# we can also perform the above task in the following way:

# import module sys to get the type of exception
import sys

randomList = ['a', 0, 2]

for entry in randomList:
    try:
        print("The entry is", entry)
        r = 1/int(entry)
        break
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        print("Next entry.")
        print()
print("The reciprocal of", entry, "is", r)

#%%
#Catching Specific Exceptions in Python
#-----------------------------------------------
#In the above example, we did not mention any specific exception in 
#the except clause.

#This is not a good programming practice as it will catch all exceptions
# and handle every case in the same way. We can specify which exceptions
# an except clause should catch.

#A try clause can have any number of except clauses to handle different 
#exceptions, however, only one will be executed in case an exception occurs.

#We can use a tuple of values to specify multiple exceptions in an 
#except clause. Here is an example pseudo code.

try:
   # do something
   pass

except ValueError:
   # handle ValueError exception
   pass

except (TypeError, ZeroDivisionError):
   # handle multiple exceptions
   # TypeError and ZeroDivisionError
   pass

except:
   # handle all other exceptions
   pass

#%%
#Raising Exceptions in Python
#-------------------------------

#In Python programming, exceptions are raised when errors occur at runtime.
# We can also manually raise exceptions using the raise keyword.
 
#We can optionally pass values to the exception to clarify why that 
#exception was raised.
    
try:
    a = int(input("Enter a positive integer: "))
    if a <= 0:
        raise ValueError("That is not a positive number!")
except ValueError as ve:
    print(ve)

#%%
#Python try with else clause
#--------------------------------
#In some situations, you might want to run a certain block of code
# if the code block inside try ran without any errors. For these
#cases, you can use the optional else keyword with the try statement.
    
#Note: Exceptions in the else clause are not handled by the 
#preceding except clauses.
    
try:
    num = int(input("Enter a number: "))
    assert num % 2 == 0
except:
    print("Not an even number!")
else:
    reciprocal = 1/num
    print(reciprocal)
    
#However, if we pass 0, we get ZeroDivisionError as the code
#block inside else is not handled by preceding except.
    
#%%
    #%%
#Python try...finally
#---------------------------
#The try statement in Python can have an optional finally clause. 
#This clause is executed no matter what, and is generally used to 
#release external resources.
    
#For example, we may be connected to a remote data center through 
#the network or working with a file or a Graphical User Interface (GUI).
    
#In all these circumstances, we must clean up the resource before 
#the program comes to a halt whether it successfully ran or not. 

#These actions (closing a file, GUI or disconnecting from network)
# are performed in the finally clause to guarantee the execution.
    
try:
   f = open("test.txt",encoding = 'utf-8')
   # perform file operations
finally:
   f.close()
   
#This type of construct makes sure that the file is closed even
# if an exception occurs during the program execution. 

#%%
#%%
#%%
   
#https://www.w3schools.com/python/python_try_except.asp






















