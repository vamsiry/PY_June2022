# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:17:43 2021

@author: rvamsikrishna
"""

#Python File I/O
#@=================    

#When we want to read from or write to a file, we need to open it first. 
#When we are done, it needs to be closed so that the resources that 
#are tied with the file are freed.

#Opening Files in Python
#============================

#Python has a built-in open() function to open a file. This function
# returns a file object, also called a handle, as it is used 
#to read or modify the file accordingly.

f = open("test.txt")    # open file in current directory
f = open("C:/Python38/README.txt")  # specifying full path

#we specify whether we want to read r, write w or append a to the file. 

#We can also specify if we want to open the file in text mode or binary mode.

#The default is reading in text mode. In this mode, we get strings
# when reading from the file.

#On the other hand, binary mode returns bytes and this is the mode 
#to be used when dealing with non-text files like images or executable files.

f = open("test.txt")      # equivalent to 'r' or 'rt'
f = open("test.txt",'w')  # write in text mode
f = open("img.bmp",'r+b') # read and write in binary mode

#the default encoding is platform dependent. 
#In windows, it is cp1252 but utf-8 in Linux.

#So, we must not also rely on the default encoding or else our 
#code will behave differently in different platforms.

f = open("test.txt", mode='r', encoding='utf-8')

#%%
#Closing Files in Python
#==================================
f = open("test.txt", encoding = 'utf-8')
# perform file operations
f.close()

#This above method is not entirely safe. If an exception occurs when
# we are performing some operation with the file, the code exits
# without closing the file.

#A safer way is to use a try...finally block.

try:
   f = open("test.txt", encoding = 'utf-8')
   # perform file operations
finally:
   f.close()
   
#This way, we are guaranteeing that the file is properly closed even 
#if an exception is raised that causes program flow to stop.
   
#The best way to close a file is by using the with statement.
# This ensures that the file is closed when the block inside 
#the with statement is exited.  
#We don't need to explicitly call the close() method. It is done internally.   

#with open("test.txt", encoding = 'utf-8') as f:
   # perform file operations
   
#%%
#Writing to Files in Python
#============================

#In order to write into a file in Python, we need to open it 
#in write w, append a or exclusive creation x mode.
   
#We need to be careful with the w mode, as it will overwrite into the 
#file if it already exists. Due to this, all the previous data are erased.
   
#Writing a string or sequence of bytes (for binary files) is done using 
#the write() method. This method returns the number of characters
# written to the file.
   
with open("test.txt",'w',encoding = 'utf-8') as f:
   f.write("my first file\n")
   f.write("This file\n\n")
   f.write("contains three lines\n")
   
#This program will create a new file named test.txt in the current 
#directory if it does not exist. If it does exist, it is overwritten.
   
#We must include the newline characters ourselves to 
#distinguish the different lines.
   
#%%
#Reading Files in Python
#========================
#To read a file in Python, we must open the file in reading r mode.
   
#We can use the read(size) method to read in the size number of data. 
#If the size parameter is not specified, it reads and returns up to 
#the end of the file.
   
#We can read the text.txt file we wrote in the above section 
#in the following way:
   
f = open("test.txt",'r',encoding = 'utf-8')
f.read(4)    # read the first 4 data
f.read(4)    # read the next 4 data
f.read()     # read in the rest till end of file
f.read()  # further reading returns empty sting

#We can read a file line-by-line using a for loop. 
#This is both efficient and fast.

for line in f:
    print(line, end = '')

#Alternatively, we can use the readline() method to read individual
# lines of a file. This method reads a file till the newline, 
#including the newline character.
    
f.readline()
f.readline()
f.readline()

#Lastly, the readlines() method returns a list of remaining lines 
#of the entire file. All these reading methods return empty values 
#   when the end of file (EOF) is reached.

f.readlines()

#%%
#%%
#%%
#Python Directory and Files Management
#=========================================
#If there are a large number of files to handle in our Python program,
#we can arrange our code within different directories to make
# things more manageable.

#A directory or folder is a collection of files and subdirectories. 
#Python has the os module that provides us with many useful methods
# to work with directories (and files as well).

#%%
#Get Current Directory
#======================
#We can get the present working directory using the getcwd() 
#We can also use the getcwdb() method to get it as bytes object.

import os
os.getcwd()  
#%%
os.getcwdb()
#%%
#The extra backslash implies an escape sequence. 
#The print() function will render this properly.
print(os.getcwd()) 

#%%
#Changing Directory
#=========================
#We can change the current working directory by using the chdir() method.
#It is safer to use an escape sequence when using the backward slash.

os.chdir('C:\\Python33')
print(os.getcwd())

#%%
#%%
#List Directories and Files
#==============================
#All files and sub-directories inside a directory can be 
#retrieved using the listdir() method.

#This method takes in a path and returns a list of subdirectories 
#and files in that path. If no path is specified, it returns the 
#list of subdirectories and files from the current working directory.
print(os.getcwd())

#%%
os.listdir()

#%%
os.listdir('G:\\')
    
#%%
#%%
#Making a New Directory
#=======================
#We can make a new directory using the mkdir() method.

#This method takes in the path of the new directory. If the full path
# is not specified, the new directory is created in the current
# working directory.

os.mkdir('test')

os.listdir()

#%%
#Renaming a Directory or a File
#=================================
#The rename() method can rename a directory or a file.

os.rename('old name','new_one name')

#%%
#Removing Directory or File
#=============================
#A file can be removed (deleted) using the remove() method.
#Similarly, the rmdir() method removes an empty directory.

os.listdir()

os.remove('old.txt')

os.rmdir('new_one')

#Note: The rmdir() method can only remove empty directories.
#In order to remove a non-empty directory, we can use the rmtree()
# method inside the shutil module.

import shutil
shutil.rmtree('test')
os.listdir()

#%%
#%%
#%%
# Note: the assert statement is used to continue the execute if the
# given condition evaluates to True. If the assert condition evaluates 
#to False, then it raises the AssertionError exception with the
# specified error message.

#%%






