# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 01:11:47 2019

@author: rvamsikrishna
"""
#https://www.hackerearth.com/practice/python/working-with-data/lists/tutorial/
#https://www.programiz.com/python-programming/list
#https://www.tutorialspoint.com/python3/python_lists.htm
#https://snakify.org/en/lessons/lists/
#https://www.w3schools.com/python/python_lists.asp



# LISTS
#==========
# list (in most programming languages the different term is used — “array”).

#List is a collection which is ordered and changeable. Allows duplicate members.

#A list contains items separated by commas and enclosed within square brackets ([])

#To some extent, lists are similar to arrays in C. One of the differences between 
#them is that all the items belonging to a list can be of different data type.

#The values stored in a list can be accessed using the slice operator ([ ] and [:]) 
#with indexes starting at 0 in the beginning of the list and working their way to end -1.

#The plus (+) sign is the list concatenation operator, and the asterisk (*) is the repetition operator.

#-----------------------------------------------------
#A[i:j]  slice j-i elements A[i], A[i+1], ..., A[j-1].

#A[i:j:-1]  slice i-j elements A[i], A[i-1], ..., A[j+1] (that is, changing 
#the order of the elements).

#A[i:j:k]  cut with the step k: A[i], A[i+k], A[i+2*k],... . If the value of k<0, 
#the elements come in the opposite order. 

#Each of the numbers i or j may be missing, what means “the beginning of line” or “the end of line" 

#Lists, unlike strings, are mutable objects: you can assign a list item to a new value.
# Moreover, it is possible to change entire slices

#%%
#Creating list
#=================
my_list = []  # empty list
my_list = [1, 2, 3] # list of integers
my_list = [1, "Hello", 3.4]  # list with mixed datatypes    

my_list = ["mouse", [8, 4, 6], ['a']] # nested list

list1 = ['physics', 'chemistry', 1997, 2000]
list2 = [1, 2, 3, 4, 5 ]
list3 = ["a", "b", "c", "d"]

print(list1)
print(list2)
print(list3)


#%%
#Accessing Values in Lists
#=============================================
list = [ 'abcd', 786 , 2.23, 'john', 70.2 ]
tinylist = [123, 'john']

print (list)          # Prints complete list
print (list[0])       # Prints first element of the list
print (list[1:3])     # Prints elements starting from 2nd till 3rd 
print (list[2:])      # Prints elements starting from 3rd element
print(list[-1])      #prints last element in the list
print(list[:])       # elements beginning to end
print(list[:-5])    # elements beginning to 4th from last

print (tinylist * 2)  # Prints list two times
print(["re"] * 3)  #Output: ["re", "re", "re"]
print (list + tinylist) # Prints concatenated 2 lists

#%%
#Nested list are accessed using nested indexing.
n_list = ["Happy", [2,0,1,5]]
print(n_list[0][1])
print(n_list[1][3])

#%%
#You can create a two-dimensional list. This is done by nesting a list inside another list.
companies = [["hackerearth", "paytm"], ["tcs", "cts"]]
print(companies)



#%%
#Basic List Operations
len([1, 2, 3])

[1, 2, 3] + [4, 5, 6]

['Hi!'] * 4

3 in [1, 2, 3]

for x in [1,2,3] : print (x,end = ' ')

#%%    
print(type([1, 2, 3]))

#%%
#Updating Lists
#==================
#You can update single or multiple elements of lists by giving the 
#slice on the left-hand side of the assignment operator, and you 
#can add to elements in a list with the append() method

list = ['physics', 'chemistry', 1997, 2000]
print ("Value available at index 2 : ", list[2])

#%%
list[2] = 2001
print ("New value available at index 2 : ", list[2])
#%%
#---------------------------
A = [1, 2, 3, 4, 5]
A[2:4] = [7, 8, 9]
print(A)
#%%
A = [1, 2, 3, 4, 5, 6, 7]
A[::-2] = [10, 20, 30, 40]
print(A)

#%%
#%%
#Loop Through a List
#=====================
#Print all items in the list, one by one:

thislist = ["apple", "banana", "cherry"]
for x in thislist:
  print(x) 

#%%
Rainbow = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
for i in range(len(Rainbow)):
    print(Rainbow[i])


#%%
#Check if Item Exists
 #======================== 
#To determine if a specified item is present in a list use the in keyword:
  
thislist = ["apple", "banana", "cherry"]

if "apple" in thislist:
  print("Yes, 'apple' is in the fruits list")   
  
  
 #%%
#List Length
#=============
#To determine how many items a list has, use the len() method:
thislist = ["apple", "banana", "cherry"]
print(len(thislist))
 
Rainbow = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
len(Rainbow)

#%%
#Add Items
#=============
# We can add one item to a list using append() method or add several items using extend() method.
#To add an item to the end of the list, use the append() method:
thislist = ["apple", "banana", "cherry"]
thislist.append("orange")
thislist.append("google")
thislist.append("facebook")
print(thislist) #Note the items are printed in the order in which they youre inserted.

#%%
thislist.extend([9, 11, 13])
print(thislist)

#%%
#we can insert one item at a desired location by using the method insert() 
#or insert multiple items by squeezing it into an empty slice of a list.

#To add an item at the specified index, use the insert() method:
thislist = ["apple", "banana", "cherry"]
thislist.insert(1, "orange")
print(thislist)

#%%
odd = [1, 9]	
odd.insert(1,3)	# Output: [1, 3, 9] 
print(odd)	
#%%
odd[2:2] = [5, 7]	# Output: [1, 3, 5, 7, 9]
print(odd)	
#%%
llist = [1,2,3,4]
thislist.extend(llist) # will add the elements in list 2 at the end of list.
print(thislist)

#%%
#Delete List Elements
#======================
#"del" statement if you know exactly which element(s) you are deleting
# remove() method if you do not know exactly which items to delete
list = ['physics', 'chemistry', 1997, 2000]
print (list)

del list[2]
print ("After deleting value at index 2 : ", list)

del list[2:4]
print ("After deleting value at index 2 : ", list)

del list
print(list)
#%%

#Remove Item
#================
#The remove() method removes the specified item:
thislist = ["apple", "banana", "cherry"]
thislist.remove("banana")
print(thislist)

#The pop() method removes the specified index, (or the last item if index is not specified):
print(thislist.pop()) # removes last value in te list
print(thislist.pop(1)) #removes value at index 1
    
#The clear() method empties the list:
print(thislist.clear()) #clear the whole list

#Finally, we can also delete items in a list by assigning an empty list to a slice of elements.
print(thislist[2:3] = [] )


#%%
#Copy a List
#================
#You cannot copy a list simply by typing list2 = list1, because: list2 will only be 
#a reference to list1, and changes made in list1 will automatically also be made in list2.

#Make a copy of a list with the copy() method:
thislist = ["apple", "banana", "cherry"]
mylist = thislist.copy()
print(mylist)

#Another way to make a copy is to use the built-in method list().
thislist = ["apple", "banana", "cherry"]
mylist = list(thislist)
print(mylist)


#%%
#List Membership Test
#========================
#We can test if an item exists in a list or not, using the keyword in.
my_list = ['p','r','o','b','l','e','m']
print('p' in my_list)
print('a' in my_list)
print('c' not in my_list)


#%%
#=========================================================================================
#============================================================================================
#The list() Constructor
#-----------------------    
#Using the list() constructor to make a List:
thislist = list(("apple", "banana", "cherry")) # note the double round-brackets

print(thislist)
type(thislist)

#Error : https://stackoverflow.com/questions/31087111/typeerror-list-object-is-not-callable-in-python

#%%    
#you can also create an empty list (the list with no items, its length is 0), 
#and you can add items to the end of your list using append
a = [] # start an empty list
n = int(input()) # read number of element in the list
for i in range(n): 
    new_element = int(input()) # read next element
    a.append(new_element) # add it to the list
    # the last two lines could be replaced by one:
    # a.append(int(input()))
print(a)    

#%%
#In the demonstrated example the empty list is created, then the number of 
#elements is read, then you read the list items line by line and append to the end. 
#The same thing can be done, saving the variable n:
a = []
for i in range(int(input())):
    a.append(int(input()))
print(a)

#%%
#There are several operations defined for lists: list concatenation (addition of lists, 
#i.e. "gluing" one list to another) and repetition (multiplying a list by a number).
a = [1, 2, 3]
b = [4, 5]
c = a + b
d = b * 3
print(c)
print(d)
print([7, 8] + [9])
print([0, 1] * 3)   

#%%
a = [0] * int(input())
for i in range(len(a)):
    a[i] = int(input())
print(a)    

#%%
#"print" displays the list items surrounded by square brackets and separated by commas   
# print all the elements in one line or one item per line

a = [1, 2, 3, 4, 5]
for i in range(len(a)):
    print(a[i])

#%%
# Here the index i is changed, then the element a[i] is displayed.
a = [1, 2, 3, 4, 5]
for elem in a:
    print(elem, end=' ')

#%%    

#for loop, which provides the convenient way to iterate over all elements of some sequence

#This is where Python differs from Pascal, where you have to iterate over elements' 
#indexes, but not over the elements themselves. 
    
#Sequences in Python are strings, lists, values of the function range() (these are not lists), 
#and some other objects. 
  
#%%
#%%    
#%%    
#for loop when you are needed to extract all the digits from a string and to make numeric list of them.
    
# given: s = 'ab12c59p7dq'
# you need to extract digits from the list s
# to make it so:
# digits == [1, 2, 5, 9, 7]
s = 'ab12c59p7dq'
digits = []
for symbol in s:
    if '1234567890'.find(symbol) != -1:
        digits.append(int(symbol))
print(digits)

#%%
'1234567890'.find("99")
#%%
#==============================================================================================
#2. Split and join methods
#========================    

#List items can be given in one line separated by a character;

#in this case, the entire list can be read using input(). 

#You can then use a string method split(), which returns a list of strings resulting 
#after cutting the initial string by spaces. Example:

# the input is a string # vamsi krishna reddy
# 1 2 3
s = input() # s == '1 2 3'
a = s.split() # a == ['1', '2', '3']
print(a)

#%%
#If you run this program with the input data of 1 2 3, the list a will be equal to 
#['1', '2', '3']. Please note that the list will consist of strings, not of numbers.

# If you want to get the list of numbers, you have to convert the list items
# one by one to integers:
a = input().split()
for i in range(len(a)):
    a[i] = int(a[i])
print(a)
print(type(a))

#%%
#Using the special magic of Python — generators — the same can be done in one line:
a = [int(s) for s in input().split()]
print(a)

# If you want to read a list of real numbers, you have to change the type int to float. 

#%%
#The method split() has an optional parameter that determines which string 
#will be used as the separator between list items.

#calling the method split('.') returns the list obtained by splitting the 
#initial string where the character '.' is encountered:

a = '192.168.0.1'.split('.')
print(a)

#%%   
#In Python, you can display a list of strings using one-line commands. 
#or that, the method join is used; this method has one parameter: a list of strings. 

#It returns the string obtained by concatenation of the elements given, and the 
#separator is inserted between the elements of the list; this separator is equal 
#to the string on which is the method applied

a = ['red', 'green', 'blue']
print(' '.join(a))
# return red green blue
print(''.join(a))
# return redgreenblue
print('***'.join(a))
# returns red***green***blue

#%%    
#If a list consists of numbers, you have to use the dark magic of generators. 
#Here's how you can print out the elements of a list, separated by spaces:
a = [1, 2, 3]

print(','.join(str(x) for x in list_of_ints))

# the next line causes a type error, as join() can only concatenate strs
#' '.join(a)

#However, if you are not a fan of dark magic, you can achieve the same effect using the loop for. 

#%%
ints = [1,2,3]
string_ints = [str(int) for int in ints] #Convert each integer to a string.
str_of_ints = ",". join(string_ints) #Combine each string with a comma.
print(str_of_ints)


#%%
#==================================================================================================
#=================================================================================================
#3. Generators
#====================

#To create a list filled with identical items, you can use the repetition of list, for example:
n = 5
a = [0] * n
print(a)

#%%
#To create more complicated lists you can use generators: the expressions 
#allowing to fill a list according to a formula.

#[expression for variable in sequence]
#where variable is the ID of some variable, 
#sequence is a sequence of values, which takes the variable (this can be a list, 
#  a string, or an object obtained using the function range
#expression — some expression, usually depending on the variable used in the generator.
# The list elements will be filled according to this expression. 
    
#This is how you create a list of n zeros using the generator:
a = [0 for i in range(5)]
print(a)

#%%
#Here's how you create a list filled with squares of integers:
n = 5
a = [i ** 2 for i in range(n)]
print(a)

#%%
# list of squares of numbers from 1 to n, you can change the settings of range to range(1, n + 1):
n = 5
a = [i ** 2 for i in range(1, n + 1)]
print(a)

#%%
#Here's how you can get a list filled with random numbers from 1 to 9 (using randrange from the module random):
from random import randrange
n = 10
a = [randrange(1, 10) for i in range(n)]
print(a)

#%%
#And in this example the list will consist of lines read from standard input:
#first,you need to enter the number of elements of the list (this value will be used as 
#an argument of the function range), second — that number of strings:
a = [input() for i in range(int(input()))]
print(a)

#%%
#===============================================================================================
#===============================================================================================
#5. Operations on lists
#============================

# x in A  --- Check whether an item in the list. Returns True or False
# x not in A  --- The same as not(x in A)
# min(A) --- The smallest element of list
# max(A) --- The largest element in the list
# A.index(x) --- The index of the first occurrence of element x in the list; in its absence generates an exception ValueError
# A.count(x) --- The number of occurrences of element x in the list 

#Built-in List Functions and Methods
#=====================================
#list functions 
#===============
# len(x)    
# list(seq) #Converts a tuple into list.
# len(my_list) #length of the list


#list methods
#===============
# list.append(obj)"----"Appends object obj to list
# list.count(obj)  "----"Returns count of how many times obj occurs in list
# list.extend(seq)"----"Appends the contents of seq to list
# list.index(obj)"----"Returns the lowest index in list that obj appears
# list.insert(index, obj)"----"Inserts object obj into list at offset index
# list.pop(obj = list[-1])"----"Removes and returns last object or obj from list
# list.remove(obj)"----"Removes object obj from list
# list.reverse()"----"Reverses objects of list in place
# list.sort([func])"----"Sorts objects of list, use compare func if given
#list.clear() "---" clear the whole list
#sort() - Sort items in a list in ascending order
#copy() - Returns a shallow copy of the list


#Some examples of Python list methods:
my_list = [3, 8, 1, 6, 0, 8, 4]
print(my_list.index(8))  # Output: 1
print(my_list.count(8))  # Output: 2
my_list.sort()
print(my_list)  # Output: [0, 1, 3, 4, 6, 8, 8]
my_list.reverse()
print(my_list)

  
#%%
#Functions over Python Lists:
#If you use another function “enumerate” over a list, it gives us a nice construct
# to get both the index and the value of the element in the list.

# loop over the companies and print both the index as youll as the name.
companies = ['hackerearth', 'google', 'facebook']

for indx, name in enumerate(companies):
    print("Index is %s for company: %s" % (indx, name))

#%%
#sorted function will sort over the list
#Similar to the sort method, you can also use the sorted function which also sorts the list
#The difference is that it returns the sorted list, while the sort method sorts the list in place.

# sort() function will modify the list it is called on. 
#The sorted() function will create a new list containing a sorted 
#version of the list it is given. 
    
#The sorted() function will not modify the list passed as a parameter.
#If you want to sort a list but still have the original 
#unsorted version, then you would use the sorted() function
    
#sorted() function will return a list so you must assign the returned
# data to a new variable. The sort() function modifies the list
# in-place and has no return value.
    
some_numbers = [4,3,5,1]
print(sorted(some_numbers))# get the sorted list
print(some_numbers) # the original list remains unchanged    

#%%
vegetables = ['squash', 'pea', 'carrot', 'potato']
new_list = sorted(vegetables)
print(new_list)

#%%
vegetables = ['squash', 'pea', 'carrot', 'potato']
print(vegetables)

#%%
vegetables.sort()

#%%
# vegetables = ['carrot', 'pea', 'potato', 'squash']
print(vegetables)

#%%

#==================================================================================================
#==================================================================================================

#%%
#List Comprehension: Elegant way to create new List
#======================================================


#List comprehensions are a tool for transforming one list (any iterable actually) into another list.

#During this transformation, elements can be conditionally included in the new list and 
#each element can be transformed as needed.

#List comprehension consists of an expression followed by for statement inside square brackets.
    
#Here is an example to make a list with each item being increasing power of 2.

pow2 = [2 ** x for x in range(10)]
print(pow2)

#This code is equivalent to
pow2 = []
for x in range(10):
  pow2.append(2 ** x)

#A list comprehension can optionally contain more for or if statements.
#An optional if statement can filter out items for the new list. Here are some examples.
#%%
pow2 = [2 ** x for x in range(10) if x > 5]
pow2
#%%
odd = [x for x in range(20) if x % 2 == 1]
odd
#%%
[x+y for x in ['Python ','C '] for y in ['Language','Programming']]

#%%
numbers = list(range(10))
doubled_odds = [n * 2 for n in numbers if n % 2 == 1]
doubled_odds
#%%
doubled_odds = map(lambda n: n * 2, filter(lambda n: n % 2 == 1, numbers))

#%%    

#From loops to comprehensions

#Every list comprehension can be rewritten as a for loop but not every for loop 
#can be rewritten as a list comprehension.

#The key to understanding when to use list comprehensions is to practice identifying 
#problems that smell like list comprehensions.

#If you can rewrite your code to look just like this for loop, you can also rewrite it
#as a list comprehension:

new_things = []
for ITEM in old_things:
  if condition_based_on(ITEM):
    new_things.append("something with " + ITEM)

#You can rewrite the above for loop as a list comprehension like this:

new_things = ["something with " + ITEM for ITEM in old_things if condition_based_on(ITEM)]

#%%

numbers = [1, 2, 3, 4, 5]

doubled_odds = []

for n in numbers:
    if n % 2 == 1:
        doubled_odds.append(n * 2)

#-------------------------
numbers = [1, 2, 3, 4, 5]
doubled_odds = [n * 2 for n in numbers if n % 2 == 1]


#%%

#Unconditional Comprehensions
#-------------------------------
doubled_numbers = []
for n in numbers:
    doubled_numbers.append(n * 2)

doubled_numbers = [n * 2 for n in numbers]

#%%
#Nested Loops
#------------------
flattened = []
for row in matrix:
    for n in row:
        flattened.append(n)

flattened = [n for row in matrix for n in row]

#When working with nested loops in list comprehensions remember that the for
#clauses remain in the same order as in our original for loops.

#%%
#Other Comprehensions
#--------------------------
#This same principle applies to set comprehensions and dictionary comprehensions.
#Code that creates a set of all the first letters in a sequence of words:
    
first_letters = set()
for w in words:
    first_letters.add(w[0])

#That same code written as a set comprehension:
first_letters = {w[0] for w in words}

#Code that makes a new dictionary by swapping the keys and values of the original one:
flipped = {}
for key, value in original.items():
    flipped[value] = key

#That same code written as a dictionary comprehension:
flipped = {value: key for key, value in original.items()}

#%%
#Readability Counts
#Remember that Python allows line breaks between brackets and braces. 

#Before
doubled_odds = [n * 2 for n in numbers if n % 2 == 1]

#After
doubled_odds = [
n * 2
for n in numbers
if n % 2 == 1
]

#%%
#Overusing list comprehensions and generator expressions in Python
#-------------------------------------------------------------------------
#This article is all about cases when comprehensions aren’t the best 
#tool for the job, at least in terms of readability.

#Writing comprehensions with poor spacing
#-------------------------------------------------
#Take the comprehension in this function:

def get_factors(dividend):
    """Return a list of all factors of the given number."""
    return [n for n in range(1, dividend+1) if dividend % n == 0]

#%%
#We could make that comprehension more readable by adding some well-placed line breaks:
def get_factors(dividend):
    """Return a list of all factors of the given number."""
    return [
            n
            for n in range(1, dividend+1)
            if dividend % n == 0
            ]
    
get_factors(10)    

#%%
#Writing ugly comprehensions
#------------------------------    
#Some loops technically can be written as comprehensions but they have so much 
#logic in them they probably shouldn’t be.

#Take this comprehension:
fizzbuzz = [
f'fizzbuzz {n}' if n % 3 == 0 and n % 5 == 0
else f'fizz {n}' if n % 3 == 0
else f'buzz {n}' if n % 5 == 0
else n
for n in range(100)
]

fizzbuzz
#%%

#This comprehension is equivalent to this for loop:
fizzbuzz = []
for n in range(100):
    fizzbuzz.append(
            f'fizzbuzz {n}' if n % 3 == 0 and n % 5 == 0
            else f'fizz {n}' if n % 3 == 0
            else f'buzz {n}' if n % 5 == 0
            else n
            )
print(fizzbuzz)
#%%
#Both the comprehension and the for loop use three nested inline if statements (Python’s ternary operator).
#Here’s a more readable way to write this code, using an if-elif-else construct:

fizzbuzz = []
for n in range(100):
    if n % 3 == 0 and n % 5 == 0:
        fizzbuzz.append(f'fizzbuzz {n}')
    elif n % 3 == 0:
        fizzbuzz.append(f'fizz {n}')
    elif n % 5 == 0:
        fizzbuzz.append(f'buzz {n}')
    else:
        fizzbuzz.append(n)

#Just because there is a way to write your code as a comprehension, that doesn’t 
#mean that you should write your code as a comprehension.
#%%    
#Be careful using any amount of complex logic in comprehensions, even a single inline if:

number_things = [
n // 2 if n % 2 == 0 else n * 3
for n in numbers
]

#If you really prefer to use a comprehension in cases like this, at least give some
#thought to whether whitespace or parenthesis could make things more readable:
number_things = [
(n // 2 if n % 2 == 0 else n * 3)
for n in numbers
]

#And consider whether breaking some of your logic out into a separate function might 
#improve readability as well (it may not in this somewhat silly example).
number_things = [
even_odd_number_switch(n)
for n in numbers
]

#Whether a separate function makes things more readable will depend on how important
#that operation is, how large it is, and how well the function name conveys the operation.

#%%
#Loops disguised as comprehensions
#------------------------------------
#Sometimes you’ll encounter code that uses a comprehension syntax but breaks the 
#spirit of what comprehensions are used for.

#For example, this code looks like a comprehension:
[print(n) for n in range(1, 11)]

#But it doesn’t act like a comprehension. We’re using a comprehension for a purpose 
#it wasn’t intended for.
[print(n) for n in range(1, 11)]

#[None, None, None, None, None, None, None, None, None, None]
#We wanted to print out all the numbers from 1 to 10 and that’s what we did. But this comprehension statement also returned a list of None values to us, which we promptly discarded.
#Comprehensions build up lists: that’s what they’re for. We built up a list of the return values from the print function and the print function returns None
#But we didn’t care about the list our comprehension built up: we only cared about its side effect.
#We could have instead written that code like this:

for n in range(1, 11):
    print(n)

#List comprehensions are for looping over an iterable and building up new lists, while for loops are for looping over an iterable to do pretty much any operation you’d like.
#When I see a list comprehension in code I immediately assume that we’re building up a new list (because that’s what they’re for).
#If you use a comprehension for a purpose outside of building up a new list, it’ll confuse others who read your code.
#If you don’t care about building up a new list, don’t use a comprehension.

#%%
#Using comprehensions when a more specific tool exists
#For many problems, a more specific tool makes more sense than a general purpose for loop.
#But comprehensions aren’t always the best special-purpose tool for the job at hand
#I have both seen and written quite a bit of code that looks like this:

import csv
with open('populations.csv') as csv_file:
    lines = [
            row
            for row in csv.reader(csv_file)
            ]

#That comprehension is sort of an identity comprehension. Its only purpose is to 
#loop over the given iterable (csv.reader(csv_file)) and create a list out of it.
#But in Python, we have a more specialized tool for this task: the list constructor
#Python’s list constructor can do all the looping and list creation work for us:

import csv
with open('populations.csv') as csv_file:
    lines = list(csv.reader(csv_file))


#Comprehensions are a special-purpose tool for looping over an iterable to build 
#up a new list while modifying each element along the way and/or filtering elements down.

#The list constructor is a special-purpose tool for looping over an iterable to
# build up a new list, without changing anything at all.

#If you don’t need to filter your elements down or map them into new elements while
# building up your new list, you don’t need a comprehension: you need the list constructor

#This comprehension converts each of the row tuples we get from looping over zip into lists

def transpose(matrix):
    """Return a transposed version of given list of lists."""
    return [
            [n for n in row]
            for row in zip(*matrix)
            ]

#We could use the list constructor for that too:
def transpose(matrix):
    """Return a transposed version of given list of lists."""
    return [
            list(row)
            for row in zip(*matrix)
            ]


#Whenever you see a comprehension like this:
my_list = [x for x in some_iterable]

#You could write this instead:
my_list = list(some_iterable)

#%%
#The same applies for dict and set comprehensions.
#This is also something I’ve written quite a bit in the past:
states = [
        ('AL', 'Alabama'),
        ('AK', 'Alaska'),
        ('AZ', 'Arizona'),
        ('AR', 'Arkansas'),
        ('CA', 'California'),
        # ...
        ]

abbreviations_to_names = {
abbreviation: name
for abbreviation, name in states
}

type(abbreviations_to_names)
abbreviations_to_names
#%%
#Here we’re looping over a list of two-item tuples and making a dictionary out of them.
#This task is exactly what the dict constructor was made for:
#abbreviations_to_names = dict(states)

#The built-in list and dict constructors aren’t the only comprehension-replacing tools.
# The standard library and third-party libraries also include tools that are sometimes better suited for your looping needs than a comprehension.

#Here’s a generator expression that sums up an iterable-of-iterables-of-numbers:

def sum_all(number_lists):
    """Return the sum of all numbers in the given list-of-lists."""
    return sum(
            n
            for numbers in number_lists
            for n in numbers
            )

a = [1,2,3,4,5,6]
sum_all(a)
#%%

#And here’s the same thing using itertools.chain:
from itertools import chain
def sum_all(number_lists):
    """Return the sum of all numbers in the given list-of-lists."""
    return sum(chain.from_iterable(number_lists))

a = [1,2,3,4,5,6]
sum_all(a)

#%%
#When you should use a comprehension and when you should use the alternative isn’t 
#always straightforward.
#I’m often torn on whether to use itertools.chain or a comprehension. I usually 
#write my code both ways and then go with the one that seems clearer.
    
#%%
#Needless work
    
#Sometimes you’ll see comprehensions that shouldn’t be replaced by another construct
# but should instead be removed entirely, leaving only the iterable they loop over.
#Here we’re opening up a file of words (with one word per line), storing file in memory, 
#and counting the number of times each occurs:

from collections import Counter
word_counts = Counter(word for word in open('word_list.txt').read().splitlines() )


#We’re using a generator expression here, but we don’t need to be. This works just as well:
from collections import Counter
word_counts = Counter(open('word_list.txt').read().splitlines())

#We were looping over a list to convert it to a generator before passing it to the Counter class. That was needless work!
#The Counter class accepts any iterable: it doesn’t care whether they’re lists, generators, tuples, or something else.
#Here’s another needless comprehension:
with open('word_list.txt') as words_file:
    lines = [line for line in words_file]
    for line in lines:
        if 'z' in line:
            print('z word', line, end='')


#We’re looping over words_file, converting it to a list of lines, and then looping over lines just once. That conversion to a list was unnecessary.
#We could just loop over words_file directly instead:
with open('word_list.txt') as words_file:
    for line in words_file:
        if 'z' in line:
            print('z word', line, end='')


#There’s no reason to convert an iterable to a list if all we’re going to do is loop over it once.
#In Python, we often care less about whether something is a list and more about whether it’s an iterable.
#Be careful not to create new iterables when you don’t need to: if you’re only going to loop over an iterable once, just use the iterable you already have.

#When would I use a comprehension?
#--------------------------------------
#So when would you actually use a comprehension?
#The simple but imprecise answer is whenever you can write your code in the below 
#comprehension copy-pasteable format and there isn’t another tool you’d rather use
# for shortening your code, you should consider using a list comprehension.

new_things = []
for ITEM in old_things:
    if condition_based_on(ITEM):
        new_things.append(some_operation_on(ITEM))


#That loop can be rewritten as this comprehension:
new_things = [
        some_operation_on(ITEM)
        for ITEM in old_things if condition_based_on(ITEM) ]

#For example here’s a for loop which doesn’t really look like it could be rewritten 
#using a comprehension:
def is_prime(candidate):
    for n in range(2, candidate):
        if candidate % n == 0:
            return False
        return True


#But there is in fact another way to write this loop using a generator expression, 
#if we know how to use the built-in "all" function:
def is_prime(candidate):
    return all(
            candidate % n != 0
            for n in range(2, candidate)
            )


#I wrote a whole article on the any and all functions and how they pair so nicely with 
#generator expressions. But any and all aren’t alone in their affinity for generator expressions.
#We have a similar situation with this code:
def sum_of_squares(numbers):
    total = 0
    for n in numbers:
        total += n**2
        return total


#There’s no append there and no new iterable being built up. But if we create a 
#generator of squares, we could pass them to the built-in sum function to get the same result:
def sum_of_squares(numbers):
    return sum(n**2 for n in numbers)    

#%%

#Any function or class that accepts an iterable as an argument might be a good candidate 
#for combining with a generator expression.
    

#%%
#Use list comprehensions thoughtfully
#-------------------------------------
#List comprehensions can make your code more readable (if you don’t believe me, see the 
#examples in my Comprehensible Comprehensions talk), but they can definitely be abused.
#https://www.youtube.com/watch?v=5_cJIcgM7rw&feature=youtu.be
    
#List comprehensions are a special-purpose tool for solving a specific problem. 
#The list and dict constructors are even more special-purpose tools for solving
# even more specific problems.
    
#Loops are a more general purpose tool for times when you have a problem that 
#doesn’t fit within the realm of comprehensions or another special-purpose looping tool.

#Functions like any, all, and sum, and classes like Counter and chain are iterable-accepting
# tools that pair very nicely with comprehensions and sometimes replace the need for 
#comprehensions entirely.
    
#Remember that comprehensions are for a single purpose: creating a new iterable from
# an old iterable, while tweaking values slightly along the way and/or for filtering 
#out values that don’t match a certain condition.

#Comprehensions are a lovely tool, but they’re not your only tool. Don’t forget the list and
# dict constructors and always consider for loops when your comprehensions get out of hand.
    








