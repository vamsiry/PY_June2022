# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:39:15 2019

@author: rvamsikrishna
"""

 
##############################################################################################
#################################################################################################
#################################################################################################    
#%%
  
#A set is a collection which is unordered and unindexed.
  
#What is a set in Python?     
#A set is an unordered collection of items. Every element is unique (no duplicates) 
#and must be immutable (which cannot be changed).

#However, the set itself is mutable. We can add or remove items from it.
  
# the order of elements in a set is undefined. You can add and delete elements of a set,
#you can iterate the elements of the set, you can perform standard operations on sets 
#(union, intersection, difference). Besides that, you can check if an element belongs to a set.
   
#Unlike arrays, where the elements are stored as ordered list, the order of elements in a 
#set is undefined (moreover, the set elements are usually not stored in order of appearance
#in the set; this allows checking if an element belongs to a set faster than just going 
#through all the elements of the set). 
  
#Any immutable data type can be an element of a set: a number, a string, a tuple. 
#Mutable (changeable) data types cannot be elements of the set
# In particular, list cannot be an element of a set (but tuple can), and another set 
#cannot be an element of a set.
#The requirement of immutability follows from the way how do computers represent sets in memory. 
  

#Sets can be used to perform mathematical set operations like union, intersection, 
#symmetric difference etc.
  
#A set is created by placing all the items (elements) inside curly braces {}, 
#separated by comma or by using the built-in function set().
 
#It can have any number of items and they may be of different types (integer, float, tuple,
#string etc.). But a set cannot have a mutable element, like list, set or dictionary, as its element.
    
# set of integers
my_set = {1, 2, 3}
print(my_set)

# set of mixed datatypes
my_set = {1.0, "Hello", (1, 2, 3)}
print(my_set)

#%%
#The order of elements is unimportant. For example, the program
A = {1, 2, 3}
B = {3, 2, 3, 1}    
print(A == B)

#Each element may enter the set only once. 
set('Hello') #returns the set of four elements: {'H', 'e', 'l', 'o'}. 



#%%
# set do not have duplicates
#--------------------------------
# Output: {1, 2, 3, 4}
my_set = {1,2,3,4,3,2}
print(my_set)

# set cannot have mutable items here [3, 4] is a mutable list
# If you uncomment line #12 nested "my_set",
# this will cause an error. #TypeError: unhashable type: 'list'

#my_set = {1, 2, [3, 4]}

# we can make set from a list
# Output: {1, 2, 3}
my_set = set([1,2,3,2])
print(my_set)

#%%
#Creating an empty set is a bit tricky.
#--------------------------------------

#Empty curly braces {} will make an empty dictionary in Python.
#To make a set without any elements we use the set() function without any argument.

# initialize a with {}
a = {}
print(type(a)) # Output: <class 'dict'>

# initialize a with set()
a = set()
print(type(a)) # Output: <class 'set'>

#%%
#Access Items
#-------------------
    
#You cannot access items in a set by referring to an index, since sets are
#unordered the items has no index.

#But you can loop through the set items using a for loop, or ask if a specified 
#value is present in a set, by using the in keyword.


#%%
#How to change a set in Python?
#-------------------------------------

#Sets are mutable. But since they are unordered, indexing have no meaning.

#We cannot access or change an element of set using indexing or slicing. Set does not support it.

#We can add single element using the add() method and multiple elements using the update() method.
#The update() method can take tuples, lists, strings or other sets as its argument. 
#In all cases, duplicates are avoided.
    
# initialize my_set
my_set = {1,3}
print(my_set)

my_set[0] # TypeError: 'set' object does not support indexing

my_set.add(2) # add an element
print(my_set)

my_set.update([2,3,4]) # add multiple elements
print(my_set)

my_set.update([4,5], {1,6,8}) # add list and set
print(my_set)

#%%
#How to remove elements from a set?
#---------------------------------------
#A particular item can be removed from set using methods, discard() and remove().

#The only difference between the two is that, while using discard() if the item does not exist
#in the set, it remains unchanged. But remove() will raise an error in such condition.

# initialize my_set
my_set = {1, 3, 4, 5, 6}
print(my_set)

# discard an element
# Output: {1, 3, 5, 6}
my_set.discard(4)
print(my_set)

# remove an element
# Output: {1, 3, 5}
my_set.remove(6)
print(my_set)

# discard an element
# not present in my_set
# Output: {1, 3, 5}
my_set.discard(2)
print(my_set)

# remove an element
# not present in my_set
# If you uncomment line 27,
# you will get an error.
# Output: KeyError: 2

#my_set.remove(2)

#%%
#Similarly, we can remove and return an item using the pop() method.
#-------------------------------------------------------------------------
#Set being unordered, there is no way of determining which item will be popped.
#It is completely arbitrary.

#We can also remove all items from a set using clear().
# initialize my_set
# Output: set of unique elements
my_set = set("HelloWorld")
print(my_set)

# pop an element
# Output: random element
print(my_set.pop())

# pop another element
# Output: random element
my_set.pop()
print(my_set)

# clear my_set
#Output: set()
my_set.clear()
print(my_set)

#The del keyword will delete the set completely:
my_set = set("HelloWorld")
del my_set
print(my_set) 


#%%
#Python Set Operations
#------------------------
#Sets can be used to carry out mathematical set operations like union, intersection, 
#difference and symmetric difference. We can do this with operators or methods.

#Union of A and B is a set of all elements from both sets.
#Union is performed using | operator. Same can be accomplished using the method union().

# initialize A and B
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

# use | operator
# Output: {1, 2, 3, 4, 5, 6, 7, 8}
print(A | B)

A.union(B)
B.union(A)

#%%
#Set Intersection
#---------------------
#Intersection of A and B is a set of elements that are common in both sets.
#Intersection is performed using & operator. Same can be accomplished using the method intersection().

# initialize A and B
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

# use & operator
# Output: {4, 5}
print(A & B)

A.intersection(B)
B.intersection(A)

#%%
#Set Difference
#---------------------
#Difference of A and B (A - B) is a set of elements that are only in A but 
#not in B. Similarly, B - A is a set of element in B but not in A.

#Difference is performed using - operator. Same can be accomplished using the method difference().

# initialize A and B
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

# use - operator on A
# Output: {1, 2, 3}
print(A - B)

A.difference(B)

B - A

B.difference(A)

#%%
#Set Symmetric Difference
#----------------------------
#Symmetric Difference of A and B is a set of elements in both A and B except 
#those that are common in both.

#Symmetric difference is performed using ^ operator. Same can be accomplished 
#using the method symmetric_difference().

# initialize A and B
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}

# use ^ operator
# Output: {1, 2, 3, 6, 7, 8}
print(A ^ B)

A.symmetric_difference(B)

B.symmetric_difference(A)

#%%
#Other Set Operations
#-------------------------
#Set Membership Test
#----------------------
# initialize my_set
my_set = set("apple")

# check if 'a' is present
# Output: True
print('a' in my_set)

# check if 'p' is present
# Output: False
print('p' not in my_set)

A = {1, 2, 3}
print(1 in A, 4 not in A)
A.add(4)

thisset = {"apple", "banana", "cherry"}
print(len(thisset)) 



#%%
#Iterating Through a Set
#---------------------------
for letter in set("apple"):
    print(letter)


#%%

#Different Python Set Methods
#----------------------------------
#add() 	Adds an element to the set
#clear() 	Removes all elements from the set
#copy() 	Returns a copy of the set
#difference() 	Returns the difference of two or more sets as a new set
#difference_update() 	Removes all elements of another set from this set
#discard() 	Removes an element from the set if it is a member. (Do nothing if the element is not in set)
#intersection() 	Returns the intersection of two sets as a new set
#intersection_update() 	Updates the set with the intersection of itself and another
#isdisjoint() 	Returns True if two sets have a null intersection
#issubset() 	Returns True if another set contains this set
#issuperset() 	Returns True if this set contains another set
#pop() 	Removes and returns an arbitary set element. Raise KeyError if the set is empty
#remove() 	Removes an element from the set. If the element is not a member, raise a KeyError
#symmetric_difference() 	Returns the symmetric difference of two sets as a new set
#symmetric_difference_update() 	Updates a set with the symmetric difference of itself and another
#union() 	Returns the union of sets in a new set
#update() 	Updates the set with the union of itself and others

#%%
#Built-in Functions with Set
#-----------------------------
#all() 	Return True if all elements of the set are true (or if the set is empty).
#any() 	Return True if any element of the set is true. If the set is empty, return False.
#enumerate() 	Return an enumerate object. It contains the index and value of all the items of set as a pair.
#len() 	Return the length (the number of items) in the set.
#max() 	Return the largest item in the set.
#min() 	Return the smallest item in the set.
#sorted() 	Return a new sorted list from elements in the set(does not sort the set itself).
#sum() 	Retrun the sum of all elements in the set.


#%%
A | B
A.union(B)
#Returns a set which is the union of sets A and B.

A |= B
A.update(B)
#Adds all elements of array B to the set A.

A & B
A.intersection(B)
#Returns a set which is the intersection of sets A and B.

A &= B
A.intersection_update(B)	
#Leaves in the set A only items that belong to the set B.

A - B
A.difference(B)
#Returns the set difference of A and B (the elements included in A, but not included in B).


A -= B
A.difference_update(B)	
#Removes all elements of B from the set A.

A ^ B
A.symmetric_difference(B)
#Returns the symmetric difference of sets A and B (the elements belonging to either A or B, but not to both sets simultaneously).

A ^= B
A.symmetric_difference_update(B)
#Writes in A the symmetric difference of sets A and B.

A <= B
A.issubset(B)
#Returns true if A is a subset of B.

A >= B
A.issuperset(B)
#Returns true if B is a subset of A.

A < B
#Equivalent to A <= B and A != B

A > B
#Equivalent to A >= B and A != B     
    

#%%
#############################################################################################
##############################################################################################
#Python Frozenset
#---------------------

#Frozenset is a new class that has the characteristics of a set, but its elements cannot 
#be changed once assigned. While tuples are immutable lists, frozensets are immutable sets.
    
#Sets being mutable are unhashable, so they can't be used as dictionary keys. On the
#other hand, frozensets are hashable and can be used as keys to a dictionary.
    
#Frozensets can be created using the function frozenset().
    
#This datatype supports methods like copy(), difference(), intersection(), isdisjoint(), 
#issubset(), issuperset(), symmetric_difference() and union(). Being immutable it does 
#not have method that add or remove elements.
    
# initialize A and B
A = frozenset([1, 2, 3, 4])
B = frozenset([3, 4, 5, 6])

A.isdisjoint(B)
A.difference(B)
A | B

frozenset({1, 2, 3, 4, 5, 6})
A.add(3)

