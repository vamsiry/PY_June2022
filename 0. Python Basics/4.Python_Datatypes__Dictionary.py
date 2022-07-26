# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:29:50 2019

@author: rvamsikrishna
"""

#Python Dictionaries
#------------------------
#Data structure, which allows to use an arbitrary type of index instead of numerical, 
#is called dictionary or associative array.

#The key in the dictionary may not be a set, but may be an element of type frozenset

#A dictionary is a set of unordered key, value pairs. In a dictionary, the keys must be
#unique and they are stored in an unordered manner.

#Let’s try to build a profile of three people using dictionaries. To do that you separate 
#the key-value pairs by a colon(“:”). The keys would need to be of an immutable type, i.e.,
# data-types for which the keys cannot be changed at runtime such as int, string, tuple, etc.
#The values can be of any type. Individual pairs will be separated by a comma(“,”) and 
#the whole thing will be enclosed in curly braces({...}).

#Dictionaries are used in the following cases: 
#------------------------------------------------
#1.to count the number of some objects. In this case, you need to make a dictionary 
#where keys are objects and values are amounts. 
#2.the storage of any data associated with the object. The keys are objects, the 
#values are associated data
#3.setting the correspondence between the objects (for instance, «the parent—descendant»).
#The key is the object and the value is the corresponding object
#4.if you need a simple array, but the maximum value of the index of the element is very large, 
#though not all the possible indexes will be used (so-called "sparse array"), you can use
#associative array to save memory. 

# Working with dictionary items
#------------------------------------
#Basic operation: getting value of the element by its key. It is written exactly as for lists: A[key].
#If there is no element with specified key in the dictionary it raises the exception KeyError.

#Another way to define the value based on a key is a method get: A.get(key). 
#If there is no element with the key get in the dictionary, it returns None.
#In the form with two arguments A.get(key, val) method returns val, if an element 
#with the key key is not in the dictionary.

#To check if an element belongs to a dictionary operations in and not in are used, same as for sets.

#To add a new item to the dictionary you just need to assign it with some value: A[key] = value.

#To remove an item from the dictionary you can use del A[key] (operation raises an 
#exception KeyError if there is no such key in the dictionary.) Here are two 
#safe ways to remove an item from the dictionary.
#-----------------------------------------------------

#Creating Dictionary
#---------------------
person_information = {'city': 'San Francisco', 'name': 'Sam', "food": "shrimps"}
type(person_information)
print(person_information)
#%%
#-----------------------
# empty dictionary
my_dict = {}
# dictionary with integer keys
my_dict = {1: 'apple', 2: 'ball'}
# dictionary with mixed keys
my_dict = {'name': 'John', 1: [2, 4, 3]}
# using dict()
my_dict = dict({1:'apple', 2:'ball'})
# from sequence having each item as a pair
my_dict = dict([(1,'apple'), (2,'ball')])

#%%
#---------------------
Capitals = {'Russia': 'Moscow', 'Ukraine': 'Kiev', 'USA': 'Washington'}
Capitals = dict(RA = 'Moscow', Ukraine = 'Kiev', USA = 'Washington')
Capitals = dict([("Russia", "Moscow"), ("Ukraine", "Kiev"), ("USA", "Washington")])
Capitals = dict(zip(["Russia", "Ukraine", "USA"], ["Moscow", "Kiev", "Washington"]))
print(Capitals)

#In the third and fourth case, you can create large dictionaries, if transferred arguments are
#a ready-made list, which can be obtained not only from by listing all the elements, 
#but are built in any other way during the execution of the program. 
#In the third way the function dict needs to recieve a list where each element
#is a tuple of two elements: key and value.
#The fourth method uses the function zip, which needs to recieve two lists of equal length: 
#a list of keys and list of values. 


#%%
#Get the values in a Dictionary
#===================================
#To get the values of a dictionary from the keys, you can directly reference the keys.
#To do this, you enclose the key in brackets [...] after writing the variable name of the dictionary.

person1_information = {'city': 'San Francisco', 'name': 'Sam', "food":"shrimps"}
print(person1_information["city"])

#%%
#You can also use the get method to retrieve the values in a dict.
#The only difference is that in the get method, you can set a default value.
#In direct referencing, if the key is not present, the interpreter throws KeyError.
alphabets = {1:'a'}

## get the value with key 1
print(alphabets.get(1)) 

#%%
# Trying to access keys which doesn't exist throws error
alphabets[2] #or

#%%
print(alphabets.get(2))  

#%%
## get the value with key 2. Pass “default” as default. Since key 2 does not exist,
# you get “default” as the return value.
print(alphabets.get(2, "default"))

#%%
## get the value with key 2 through direct referencing
print(alphabets[2])

#%%
dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
print ("dict['Name']: ", dict['Name'])
print ("dict['Age']: ", dict['Age'])

#%%
#Change or Add elements to a dictionary
#===============================     
#Dictionary are mutable. We can add new items or change the value of existing
# items using assignment operator.
     
#If the key is already present, value gets updated,
# else a new key: value pair is added to the dictionary.     
     
# initialize an empty dictionary
person1_information = {}

# add the key, value information with key “city”
person1_information["city"] = "San Francisco"
# print the present person1_information
print(person1_information)


# add another key, value information with key “name”
person1_information["name"] = "Sam"
# print the present dictionary
print(person1_information)

# add another key, value information with key “food”
person1_information["food"] = "shrimps"
# print the present dictionary
print(person1_information)

#%%
#Or you can combine two dictionaries to get a larger dictionary using the update method.
#-------------------------------------------------------------------------------------
# create a small dictionary
person1_information = {'city': 'San Francisco'}
# print it and check the present elements in the dictionary
print(person1_information) 


# have a different dictionary
remaining_information = {'name': 'Sam', "food": "shrimps"}


# add the second dictionary remaining_information to personal1_information using the update method
person1_information.update(remaining_information)
# print the current dictionary
print(person1_information)

#%%
thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
thisdict["year"] = 2018

print(thisdict)

#%%
#Delete or remove elements of a dictionary
#======================================

#pop(), This method removes as item with the provided key and returns the value.

#popitem() can be used to remove and return an arbitrary item (key, value) form the dictionary.

#All the items can be removed at once using the clear() method.

#also use the del keyword to remove individual items(key, value pair) or the entire dictionary itself.

#The popitem() method removes the last inserted item (in versions before 3.7, 
#a random item is removed instead):


person1_information = {'city': 'San Francisco', 'name': 'Sam', "food": "shrimps"}
# delete the key, value pair with the key “food”
del person1_information["food"]
print(person1_information)

#%%
#A disadvantage is that it gives KeyError if you try to delete a nonexistent key.
del person1_information["non_existent_key"]

#%%
#So, instead of the del statement you can use the pop method. This method takes in the key 
#as the parameter. As a second argument, you can pass the default value if the key is not present.
print(person1_information.pop("food", None))
print(person1_information)
print(person1_information.pop("food", None))

#%%
# create a dictionary
squares = {1:1, 2:4, 3:9, 4:16, 5:25}  

# remove a particular item
# Output: 16
print(squares.pop(4))  
print(squares) # Output: {1: 1, 2: 4, 3: 9, 5: 25}

#%%
# remove an arbitrary item # Output: (1, 1)
print(squares.popitem())
# Output: {2: 4, 3: 9, 5: 25}
print(squares)

#%%
# delete a particular item
del squares[5]  
# Output: {2: 4, 3: 9}
print(squares)
#%%
# remove all items
squares.clear()
# Output: {}
print(squares)

#%%
# delete the dictionary itself
del squares
#print(squares) # Throws Error

#%%
#Copy a Dictionary
#===================
#You cannot copy a dictionary simply by typing dict2 = dict1,
# because: dict2 will only be a reference to dict1, and changes made in dict1 will 
#automatically also be made in dict2.

#There are ways to make a copy, one way is to use the built-in Dictionary method copy().

thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
mydict = thisdict.copy()
print(mydict)

#%%
#Make a copy of a dictionary with the dict() method:
thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
mydict = dict(thisdict)
print(mydict) 

#%%
#%%
#More facts about the Python dictionary
#=========================================
#You can test the presence of a key using the has_key method.

alphabets = {1: 'a'}
print(1 in alphabets)
print(2 in alphabets)

#%%
#A dictionary in Python doesn't preserve the order. Hence, we get the following:
call = {'sachin': 4098, 'guido': 4139}
call["snape"] = 7663
call


#%%
#%%
#Python Dictionary Methods
#==============================
#clear() 	Remove all items form the dictionary.
#copy() 	Return a shallow copy of the dictionary.
#fromkeys(seq[, v]) 	Return a new dictionary with keys from seq and value equal to v (defaults to None).

#get(key[,d]) 	Return the value of key. If key doesnot exit, return d (defaults to None).
#items() 	Return a new view of the dictionary's items (key, value).
#keys() 	Return a new view of the dictionary's keys.
#pop(key[,d]) 	Remove the item with key and return its value or d if key is not found. If d is not provided and key is not found, raises KeyError.
#popitem() 	Remove and return an arbitary item (key, value). Raises KeyError if the dictionary is empty.
#setdefault(key[,d]) 	If key is in the dictionary, return its value. If not, insert key with a value of d and return d (defaults to None).
#update([other]) 	Update the dictionary with the key/value pairs from other, overwriting existing keys.
#values() 	Return a new view of the dictionary's values

marks = {}.fromkeys(['Math','English','Science'], 0)
print(marks) # Output: {'English': 0, 'Math': 0, 'Science': 0}

#%%
for item in marks.items():
    print(item)
#%%
list(sorted(marks.keys())) # Output: ['English', 'Math', 'Science']


#%%
#Built-in Functions with Dictionary
#====================================
#all() 	Return True if all keys of the dictionary are true (or if the dictionary is empty).
#any() 	Return True if any key of the dictionary is true. If the dictionary is empty, return False.
#len() 	Return the length (the number of items) in the dictionary.
#cmp() 	Compares items of two dictionaries.
#sorted() 	Return a new sorted list of keys in the dictionary.
#str(dict)   Produces a printable string representation of a dictionary

squares = {1: 1, 3: 9, 5: 25, 7: 49, 9: 81}
print(len(squares))
print(sorted(squares)) # Output: [1, 3, 5, 7, 9]

#%%
#Python Dictionary Comprehension
#=====================================
#Dictionary comprehension is an elegant and concise way to create new dictionary
# from an iterable in Python.

squares = {x: x*x for x in range(6)}
print(squares) # Output: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

#This code is equivalent to
squares = {}

for x in range(6):
    squares[x] = x*x

#A dictionary comprehension can optionally contain more for or if statements.
#An optional if statement can filter out items to form the new dictionary.

odd_squares = {x: x*x for x in range(11) if x%2 == 1}
print(odd_squares) # Output: {1: 1, 3: 9, 5: 25, 7: 49, 9: 81}


#%%
#Other Dictionary Operations
#===========================

#Dictionary Membership Test
#----------------------------
#We can test if a key is in a dictionary or not using the keyword in. 
#Notice that membership test is for keys only, not for values.

squares = {1: 1, 3: 9, 5: 25, 7: 49, 9: 81}

print(1 in squares)
print(2 not in squares)
print(49 in squares)

#%%

#Iterating or Looping over dictionary
#------------------------------------
#Say, you got a dictionary, and you want to print the keys and values in it. Note that
#the key-words for and in are used which are used when you try to loop over something. 

person1_information = {'city': 'San Francisco', 'name': 'Sam', "food": "shrimps"}

for k, v in person1_information.items():
     print("key is: %s" % k)
     print("value is: %s" % v)
     print("----------------------------")


#---------------------------------------------------
squares = {1: 1, 3: 9, 5: 25, 7: 49, 9: 81}
for i in squares:
    print(squares[i])
   

#-----------------------------------------------------    
# Create empty dict Capitals
Capitals = dict()
# Fill it with some values
Capitals['Russia'] = 'Moscow'
Capitals['Ukraine'] = 'Kiev'
Capitals['USA'] = 'Washington'
print(Capitals)

Countries = ['Russia', 'France', 'USA', 'Russia']

for country in Countries:
  # For each country from the list check to see whether it is in the dictionary Capitals
    if country in Capitals:
        print('The capital of ' + country + ' is ' + Capitals[country])
    else:
        print('The capital of ' + country + ' is unknown')    
        
               
#----------------------------------------------------------------
A = {'ab' : 'ba', 'aa' : 'aa', 'bb' : 'bb', 'ba' : 'ab'}

key = 'ac'
if key in A:
    del A[key]

try:
    del A[key]
except KeyError:
	print('There is no element with key "' + key + '" in dict')
print(A)        
   
#-----------------------------------------------------------------
A = dict(zip('abcdef', list(range(6))))
print(A)
for key in A:
    print(key, A[key])

#----------------------------------------------------------------
A = dict(zip('abcdef', list(range(6))))
for key, val in A.items():
    print(key, val)
    
    
#----------------------------------------------------------------
thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
    
for x in thisdict:
  print(x) 

#Print all values in the dictionary, one by one:
for x in thisdict:
  print(thisdict[x])     
    
#You can also use the values() function to return values of a dictionary:
for x in thisdict.values():
  print(x) 

#Loop through both keys and values, by using the items() function:
for x, y in thisdict.items():
  print(x, y) 

    
    

#%%
#%%
#Properties of Dictionary Keys
#------------------------------
#Dictionary values have no restrictions. They can be any arbitrary Python object, 
#either standard objects or user-defined objects. However, same is not true for the keys.

#1.no duplicate key is allowed. When duplicate keys are encountered during assignment, 
#the last assignment wins
dict = {'Name': 'Zara', 'Age': 7, 'Name': 'Manni'}
print ("dict['Name']: ", dict['Name'])

#2.Keys must be immutable. This means you can use strings, numbers or tuples as 
#dictionary keys but something like ['key'] is not allowed
dict = {['Name']: 'Zara', 'Age': 7}
print ("dict['Name']: ", dict['Name'])

    
    
#%%
#%%    
#%%    
#Nested Dictionary in Python?
#=============================

#a nested dictionary is a dictionary inside a dictionary.
# It's a collection of dictionaries into one single dictionary.
    
    
#Create a Nested Dictionary
#------------------------------
people = {1: {'name': 'John', 'age': '27', 'sex': 'Male'},
          2: {'name': 'Marie', 'age': '22', 'sex': 'Female'}}

print(people)

#%%
#Access elements of a Nested Dictionary
#------------------------------------------
people = {1: {'name': 'John', 'age': '27', 'sex': 'Male'},
          2: {'name': 'Marie', 'age': '22', 'sex': 'Female'}}

print(people[1]['name'])
print(people[1]['age'])
print(people[1]['sex'])

#%%
#Add element to a Nested Dictionary
#----------------------------------------
people = {1: {'name': 'John', 'age': '27', 'sex': 'Male'},
          2: {'name': 'Marie', 'age': '22', 'sex': 'Female'}}

people[3] = {}

people[3]['name'] = 'Luna'
people[3]['age'] = '24'
people[3]['sex'] = 'Female'
people[3]['married'] = 'No'

print(people[3])


people = {1: {'name': 'John', 'age': '27', 'sex': 'Male'},
          2: {'name': 'Marie', 'age': '22', 'sex': 'Female'},
          3: {'name': 'Luna', 'age': '24', 'sex': 'Female', 'married': 'No'}}

people[4] = {'name': 'Peter', 'age': '29', 'sex': 'Male', 'married': 'Yes'}
print(people[4])

#%%
#Delete elements from a Nested Dictionary
#------------------------------------------
people = {1: {'name': 'John', 'age': '27', 'sex': 'Male'},
          2: {'name': 'Marie', 'age': '22', 'sex': 'Female'},
          3: {'name': 'Luna', 'age': '24', 'sex': 'Female', 'married': 'No'},
          4: {'name': 'Peter', 'age': '29', 'sex': 'Male', 'married': 'Yes'}}

del people[3]['married']
del people[4]['married']

print(people[3])
print(people[4])

#%%
#delete dictionary from a nested dictionary?
#--------------------------------------------------------
people = {1: {'name': 'John', 'age': '27', 'sex': 'Male'},
          2: {'name': 'Marie', 'age': '22', 'sex': 'Female'},
          3: {'name': 'Luna', 'age': '24', 'sex': 'Female'},
          4: {'name': 'Peter', 'age': '29', 'sex': 'Male'}}

del people[3], people[4]
print(people)

#%%
#Iterating Through a Nested Dictionary
#-----------------------------------------
people = {1: {'Name': 'John', 'Age': '27', 'Sex': 'Male'},
          2: {'Name': 'Marie', 'Age': '22', 'Sex': 'Female'}}

for p_id, p_info in people.items():
    print("\nPerson ID:", p_id)
    for key in p_info:
        print(key + ':', p_info[key])    

#Key Points to Remember:
#--------------------------
#Nested dictionary is an unordered collection of dictionary
#Slicing Nested Dictionary is not possible.
#We can shrink or grow nested dictionary as need.
#Like Dictionary, it also has key and value.
#Dictionary are accessed using key.
        
#%%





