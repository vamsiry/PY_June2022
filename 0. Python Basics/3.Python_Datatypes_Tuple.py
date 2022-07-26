# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 22:52:02 2019

@author: rvamsikrishna
"""

#A tuple in Python is similar to a list. The difference between the two is that 
#we cannot change the elements of a tuple once it is assigned whereas, in a list, 
#elements can be changed

#Advantages of Tuple over List
#--------------------------------
#We generally use tuple for heterogeneous (different) datatypes and list for 
#homogeneous (similar) datatypes.

#Since tuples are immutable, iterating through tuple is faster than with list. 
#So there is a slight performance boost.

#Tuples that contain immutable elements can be used as a key for a dictionary. 
#With lists, this is not possible.

#If you have data that doesn't change, implementing it as tuple will guarantee 
#that it remains write-protected.



#Creating a Tuple
#-------------------
#A tuple is created by placing all the items (elements) inside parentheses (), 
#separated by commas. The parentheses are optional, however, it is a good practice to use them.

#A tuple can have any number of items and they may be of different types (integer, float, list, string, etc.).

# Empty tuple
my_tuple = ()
print(my_tuple)  # Output: ()

# Tuple having integers
my_tuple = (1, 2, 3)
print(my_tuple)  # Output: (1, 2, 3) 

# tuple with mixed datatypes
my_tuple = (1, "Hello", 3.4)
print(my_tuple)  # Output: (1, "Hello", 3.4)  

# nested tuple
my_tuple = ("mouse", [8, 4, 6], (1, 2, 3)) 
print(my_tuple) # Output: ("mouse", [8, 4, 6], (1, 2, 3)) 

#The tuple() Constructor
#It is also possible to use the tuple() constructor to make a tuple.
thistuple = tuple(("apple", "banana", "cherry")) # note the double round-brackets
print(thistuple)

#%%
#A tuple can also be created without using parentheses. This is known as tuple packing.

my_tuple = 3, 4.6, "dog"
print(my_tuple)   # Output: 3, 4.6, "dog" 
print(type(my_tuple))

#%%
# tuple unpacking is also possible
a, b, c = my_tuple

print(a)      # 3
print(b)      # 4.6 
print(c)      # dog 

#%%
#Creating a tuple with one element is a bit tricky.
#Having one element within parentheses is not enough. We will need a trailing comma
# to indicate that it is, in fact, a tuple.

my_tuple = ("hello")
print(type(my_tuple))  # <class 'str'>

# Creating a tuple having one element
my_tuple = ("hello",)  
print(type(my_tuple))  # <class 'tuple'> 

# Parentheses is optional
my_tuple = "hello",
print(type(my_tuple))  # <class 'tuple'> 

#%%
#Access Tuple Elements
#---------------------
#1. Indexing

#We can use the index operator [] to access an item in a tuple
#Likewise, nested tuples are accessed using nested indexing,

my_tuple = ('p','e','r','m','i','t')

print(my_tuple[0])   # 'p' 
print(my_tuple[5])   # 't'

# IndexError: list index out of range
# print(my_tuple[6])

# Index must be an integer
# TypeError: list indices must be integers, not float
# my_tuple[2.0]

# nested tuple
n_tuple = ("mouse", [8, 4, 6], (1, 2, 3))

# nested index
print(n_tuple[0][3])       # 's'
print(n_tuple[1][1])       # 4

#%%
#2. Negative Indexing

my_tuple = ('p','e','r','m','i','t')

# Output: 't'
print(my_tuple[-1])

# Output: 'p'
print(my_tuple[-6])

#%%
#3. Slicing

my_tuple = ('p','r','o','g','r','a','m','i','z')

# elements 2nd to 4th
# Output: ('r', 'o', 'g')
print(my_tuple[1:4])

# elements beginning to 2nd
# Output: ('p', 'r')
print(my_tuple[:-7])

# elements 8th to end
# Output: ('i', 'z')
print(my_tuple[7:])

# elements beginning to end
# Output: ('p', 'r', 'o', 'g', 'r', 'a', 'm', 'i', 'z')
print(my_tuple[:])    

#%%
#Changing a Tuple
#-----------------
#Unlike lists, tuples are immutable.
#This means that elements of a tuple cannot be changed once it has been assigned. 
#But, if the element is itself a mutable datatype like list, its nested items can be changed.

#We can also assign a tuple to different values (reassignment).

my_tuple = (4, 2, 3, [6, 5])


# TypeError: 'tuple' object does not support item assignment
# my_tuple[1] = 9

# However, item of mutable element can be changed
my_tuple[3][0] = 9    # Output: (4, 2, 3, [9, 5])
print(my_tuple)

# Tuples can be reassigned
my_tuple = ('p','r','o','g','r','a','m','i','z')

# Output: ('p', 'r', 'o', 'g', 'r', 'a', 'm', 'i', 'z')
print(my_tuple)


#%%

#We can use + operator to combine two tuples. This is also called concatenation.
#We can also repeat the elements in a tuple for a given number of times using the * operator.
#Both + and * operations result in a new tuple.

# Concatenation
#--------------------
# Output: (1, 2, 3, 4, 5, 6)
print((1, 2, 3) + (4, 5, 6))

# Repeat
# Output: ('Repeat', 'Repeat', 'Repeat')
print(("Repeat",) * 3)


tup1 = (12, 34.56)
tup2 = ('abc', 'xyz')
tup3 = tup1 + tup2
print (tup3)


#%%
#Deleting a Tuple
#-----------------------
#we cannot change the elements in a tuple. That also means we cannot delete or remove items from a tuple.
#But deleting a tuple entirely is possible using the keyword del.

my_tuple = ('p','r','o','g','r','a','m','i','z')

# can't delete items
# TypeError: 'tuple' object doesn't support item deletion
# del my_tuple[3]

# Can delete an entire tuple
del my_tuple

# NameError: name 'my_tuple' is not defined
print(my_tuple)

#%%

#Tuple Methods

#Methods that add items or remove items are not available with tuple. 
#Only the following two methods are available.

#count(x) 	Returns the number of items x
#index(x) 	Returns the index of the first item that is equal to x

my_tuple = ('a','p','p','l','e',)

print(my_tuple.count('p'))  # Output: 2
print(my_tuple.index('l'))  # Output: 3

#%%
#Other Tuple Operations

#1. Tuple Membership Test

#We can test if an item exists in a tuple or not, using the keyword in.
my_tuple = ('a','p','p','l','e',)

# In operation
# Output: True
print('a' in my_tuple)

# Output: False
print('b' in my_tuple)

# Not in operation
# Output: True
print('g' not in my_tuple)

len((1, 2, 3))
(1, 2, 3) + (4, 5, 6)
('Hi!',) * 4
3 in (1, 2, 3)
for x in (1,2,3) : print (x, end = ' ') #Iteration

#Built-in Tuple Functions
#cmp(tuple1, tuple2)  Compares elements of both tuples.
#tuple(seq)  Converts a list into tuple.


#%%
#2. Iterating Through a Tuple

#Using a for loop we can iterate through each item in a tuple.
# Output: 
# Hello John
# Hello Kate
for name in ('John','Kate'):
     print("Hello",name)    
     
     
thistuple = ("apple", "banana", "cherry")
for x in thistuple:
  print(x) 
  
  
#Check if Item Exists
thistuple = ("apple", "banana", "cherry")
if "apple" in thistuple:
  print("Yes, 'apple' is in the fruits tuple") 

  
#%%




































