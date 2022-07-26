# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:17:28 2019

@author: rvamsikrishna
"""

#https://www.geeksforgeeks.org/python-counting-sign-change-in-list-containing-positive-and-negative-integers/
       

#%%        
#%%
#%%
# Python program to print even numbers in a list
#-----------------------------------------------------------------
list1 = [10, 21, 4, 44, 66, 93] 

for num in list1: 
    if num % 2 == 0: 
       print(num, end = " ") 
#%%
#Using while loop :(value of even indexes)
#-----------------
list1 = [10, 21, 4, 44, 66, 93] 

num = 0
while(num < len(list1)): 
    if num % 2 == 0: 
       print(list1[num], end = " ") 
    num += 1
    
#%%
list1 = [10, 21, 4, 44, 66, 93] 

num = 0
while(num < len(list1)): 
    if list1[num] % 2 == 0: 
       print(list1[num], end = " ") 
    num += 1
    
#%%    
#Using list comprehension : 
#------------------------------
list1 = [10, 21, 4, 45, 66, 93] 
  
# using list comprehension 
even_nos = [num for num in list1 if num % 2 == 0] 
  
print("Even numbers in the list: ", even_nos) 

#%%
#Using lambda expressions :
#--------------------------
list1 = [10, 21, 4, 45, 66, 93, 11,22]  
  
# we can also print even no's using lambda exp.  
even_nos = list(filter(lambda x: (x % 2 == 0), list1)) 
  
print("Even numbers in the list: ", even_nos)    


#%%
#%%
#%%
# Python program to print odd numbers  in a List
#-------------------------------------------------
list1 = [10, 21, 4, 45, 66, 93] 
  
# iterating each number in list 
for num in list1: 
    if num % 2 != 0: 
       print(num, end = " ") 
       
#%%
#Using while loop : (odd indexed num)
#--------------------
list1 = [10, 21, 4, 45, 66, 93] 

num = 0  
# using while loop         
while(num < len(list1)): 
    if num % 2 != 0: 
       print(list1[num], end = " ") 
    num += 1       
    
#%%
list1 = [10, 21, 4, 45, 66, 93] 

num = 0  
# using while loop         
while(num < len(list1)): 
    if list1[num] % 2 != 0: 
       print(list1[num], end = " ") 
    num += 1       
    

#%%
#Using list comprehension : 
#------------------------------
list1 = [10, 21, 4, 45, 66, 93] 
  
only_odd = [num for num in list1 if num % 2 == 1] 
  
print(only_odd) 

#%%
#Using lambda expressions :       
#----------------------------
list1 = [10, 21, 4, 45, 66, 93, 11]  
   
odd_nos = list(filter(lambda x: (x % 2 != 0), list1)) 
  
print("Odd numbers in the list: ", odd_nos)  


#%%
#%%
#%%
# Python program to print negative numbers in a list
#-----------------------------------------------------------
list1 = [11, -21, 0, 45, 66, -93] 
# iterating each number in list 
for num in list1: 
    if num < 0: 
       print(num, end = " ") 

#%%
# Example #2: Using while loop
list1 = [-10, 21, -4, -45, -66, 93] 
num = 0  
# using while loop      
while(num < len(list1)): 
    if list1[num] < 0: 
        print(list1[num], end = " ") 
    num += 1
            
 #%%   
# Example #3: Using list comprehension
list1 = [-10, -21, -4, 45, -66, 93] 
# using list comprehension 
neg_nos = [num for num in list1 if num < 0] 
print("Negative numbers in the list: ", *neg_nos)        

#%%
#Example #4: Using lambda expressions
list1 = [-10, 21, 4, -45, -66, 93, -11]  
# we can also print negative no's using lambda exp.  
neg_nos = list(filter(lambda x: (x < 0), list1))   
print("Negative numbers in the list: ", *neg_nos)  


#%%
#%%
#%%
# Python | Create list of numbers with given range
#------------------------------------------------------
# Approach #1 : Naive Appraoch 
#------------------------------
def createList(r1, r2): 
      if (r1 == r2): 
        return r1 
      else: 
        res = [] 
        while(r1 < r2+1 ): 
            res.append(r1) 
            r1 += 1
        return res 
      
# Driver Code 
#r1, r2 = -1, 1
r1, r2 = 5, 1
print(createList(r1, r2)) 

#%%
def createList(r1, r2):      
      if (r1 == r2): 
        return r1 
      else: 
        res = []
        if (r1 < r2):            
            while(r1 < r2+1 ): 
                res.append(r1) 
                r1 += 1
        else :
            while(r1 >= r2 ): 
                res.append(r1) 
                r1 -= 1
        return res 
      
# Driver Code 
#r1, r2 = -1, 1
r1, r2 = 5, 1
print(createList(r1, r2)) 

#%%
# Approach #2 : List comprehension
#---------------------------------------
def createList(r1, r2): 
    return [item for item in range(r1, r2+1)] 
      
# Driver Code 
r1, r2 = -1, 1
#r1, r2 = 5, 1

print(createList(r1, r2)) 

#%%
# Approach #3 : using Python range()
#-----------------------------------------
def createList(r1, r2): 
    return list(range(r1, r2+1)) 
      
# Driver Code 
r1, r2 = -1, 1
print(createList(r1, r2)) 

#%%
# Approach #4 : Using numpy.arange()
#---------------------------------------
import numpy as np 
def createList(r1, r2): 
    return np.arange(r1, r2+1, 1) 
      
# Driver Code 
r1, r2 = -1, 1
print(createList(r1, r2)) 


#%%
#%%
#%%
#Python | count of Numbers in a list within a given range
#------------------------------------------------------
def count(list1, l, r): 
    c = 0 
    for x in list1: 
        if x>= l and x<= r: 
            c+= 1 
    return c 
      
# driver code 
list1 = [10, 20, 30, 40, 50, 40, 40, 60, 70] 
l = 40
r = 80 
print(count(list1, l, r)) 

#%%
#Single Line Approach:
#------------------------

def count(list1, l, r): 
    return len(list(x for x in list1 if l <= x <= r)) 
  
# driver code 
list1 = [10, 20, 30, 40, 50, 40, 40, 60, 70] 
l = 40
r = 80 
print(count(list1, l, r) )


#%%
#%%
#%%
#Python program to print all even numbers in a range
#---------------------------------------------------------

#Example #1: Print all even numbers from given list using for loop

#start = int(input("Enter the start of range: ")) 
#end = int(input("Enter the end of range: ")) 

start, end = 4, 19
  
for num in range(start, end + 1): 
    if num % 2 == 0: 
        print(num, end = " ") 
        

#%%
#%%
#%%
#Python program to print all odd numbers in a range
#---------------------------------------------------
#start, end = 4, 19

start = int(input("Enter the start of range: ")) 
end = int(input("Enter the end of range: ")) 
      
for num in range(start, end + 1): 
    if num % 2 != 0: 
        print(num, end = " ")         
    
       
#%%
#%%
#%%
# Python program to print all negative numbers in a range
#----------------------------------------------------------------
start = int(input("Enter the start of range: ")) 
end = int(input("Enter the end of range: ")) 
  
for num in range(start, end + 1): 
    if num < 0: 
        print(num, end = " ")  

#%%
#%%
#%%
#Python program to count of positive and negative numbers in a list
#------------------------------------------------------------------
list1 = [10, -21, 4, -45, 66, -93, 1] 
pos_count, neg_count = 0, 0
  
# iterating each number in list 
for num in list1: 
    if num >= 0: 
        pos_count += 1
    else: 
        neg_count += 1
          
print("Positive numbers in the list: ", pos_count) 
print("Negative numbers in the list: ", neg_count)         

#%%
#Example #2: Using while loop
#-----------------------------

list1 = [-10, -21, -4, -45, -66, 93, 11]   
pos_count, neg_count = 0, 0
num = 0
  
# using while loop      
while(num < len(list1)): 
    if list1[num] >= 0: 
        pos_count += 1
    else: 
        neg_count += 1
    num += 1
    
print("Positive numbers in the list: ", pos_count) 
print("Negative numbers in the list: ", neg_count)         
 
#%%
#Example #3 : Using Python Lambda Expressions
#-----------------------------------------------
list1 = [10, -21, -4, 45, 66, 93, -11] 
  
neg_count = len(list(filter(lambda x: (x < 0), list1))) 
  
# we can also do len(list1) - neg_count 
pos_count = len(list(filter(lambda x: (x >= 0), list1))) 
  
print("Positive numbers in the list: ", pos_count) 
print("Negative numbers in the list: ", neg_count) 

#%%
#Example #4 : Using List Comprehension
#--------------------------------------
list1 = [-10, -21, -4, -45, -66, -93, 11] 
  
only_pos = [num for num in list1 if num >= 1] 
pos_count = len(only_pos) 
  
print("Positive numbers in the list: ", pos_count) 
print("Negative numbers in the list: ", len(list1) - pos_count) 

#%%
#%%
#%%
#Counting sign change in list containing Positive and Negative Integers
#--------------------------------------------------------------------------
#https://www.geeksforgeeks.org/python-counting-sign-change-in-list-containing-positive-and-negative-integers/

#Method #1: Using Iteration
#-----------------------------
Input = [-10, 2, 3, -4, 5, -6, 7, 8, -9, 10,0,0,0, -11, 12]  
  
# Variable Initialization 
prev = int(Input[0]/Input[0])
ans = 0
  
# Using Iteration 
for elem in Input: 
    if elem == 0: 
        sign = -1
    else: 
        sign = elem / abs(elem) 
    if sign == -prev: 
        ans = ans + 1
        prev = sign 
  
# Printing answer 
print(ans) 


#%%
#Method #2: Using Itertools and groupby
#-----------------------------------------
Input = [-1, 2, 3, -4, 5, -6, 7, 8, -9, 10, -11, 12]  
  
# Importing 
import itertools 
  
# Using groupby 
Output = len(list(itertools.groupby(Input, lambda Input: Input > 0)))  
Output = Output -1
print(Output) 

#%%
Input = [-1, 2, 3, -4, 5, -6, 7, 8, -9, 10, -11, 12]  

import itertools 

Output = list(itertools.groupby(Input, lambda Input: Input > 0))

for key, group in Output: 
    key_and_group = {key : list(group)} 
    print(key_and_group) 

#%%
#https://www.geeksforgeeks.org/itertools-groupby-in-python/    
Input = [-1, 2, 3, -4, 5, -6, 7, 8, -9, 10, -11, 12]  

import itertools 
    
for key, group in itertools.groupby(Input, lambda Input: Input > 0): 
    print(" :", list(group)) 

#%%
#Method #3: Using Zip
#------------------------
def check(Input): 
    Input = [-1 if not x else x for x in Input] 
    # zip with leading 1, so that opening negative value is  
    # treated as sign change 
    return sum((x ^ y)<0 for x, y in zip([1]+Input, Input)) 
  
# Input list Initialization 
Input = [-10, 2, 3, -4, 5, -6, 7, 8, -9, 10, -11, 12]  
Output = check(Input) 
  
Output = Output -1
  
# Printing output 
print(Output) 


#%%
#%%
#%%
#Python | Ways to create triplets from given list
#-------------------------------------------------

#Method #1: Using List comprehension 
#-------------------------------------
# List of word initialization 
list_of_words = ['I', 'am', 'Paras', 'Jain', 
                 'I', 'Study', 'DS', 'Algo'] 
  
# Using list comprehension 
List = [list_of_words[i:i+3]  for i in range(len(list_of_words) - 2)] 
  
print(List) 

#%%
# Method #2: Using Iteration 
#-------------------------------
# List of word initialization 
list_of_words = ['I', 'am', 'Paras', 'Jain', 
                 'I', 'Study', 'DS', 'Algo'] 

# Output list initialization 
out = [] 

length = len(list_of_words) 
  
# Using iteration 
for z in range(0, length-2): 
    # Creating a temp list to add 3 words 
    temp = [] 
    temp.append(list_of_words[z]) 
    temp.append(list_of_words[z + 1]) 
    temp.append(list_of_words[z + 2]) 
    out.append(temp) 
  
# printing output 
print(out) 



#%%
#%%
#%%
#Python | Ways to create a dictionary of Lists
#------------------------------------------------
#https://www.geeksforgeeks.org/python-ways-to-create-a-dictionary-of-lists/

#Note that the restriction with keys in Python dictionary is 
#only immutable data types can be used as keys, which means we
# cannot use a dictionary of list as a key. 


# Creating a dictionary 
myDict = {[1, 2]: 'Geeks'} 
print(myDict) 

#Output: TypeError: unhashable type: 'list'

#%%
#But the same can be done very wisely with values in dictionary
#Let’s see all the different ways we can create a dictionary of Lists.

#Method #1: Using subscript
#--------------------------
myDict = {} 
  
# Adding list as value 
myDict["key1"] = [1, 2] 
myDict["key2"] = ["Geeks", "For", "Geeks"]  
  
print(myDict) 

#%%
# Method #2: Adding nested list as value using append() method.
#---------------------------------------------------------------
myDict = {} 
  
myDict["key1"] = [1, 2] 
  
lst = ['Geeks', 'For', 'Geeks'] 
  
# Adding this list as sublist in myDict 
myDict["key1"].append(lst) 
  
print(myDict) 

#%%
# Method #3: Using setdefault() method
#-----------------------------------------
#Iterate the list and keep appending the elements till given 
#range using setdefault() method.

myDict = dict() 
  
# Creating a list 
valList = ['1', '2', '3'] 
  
# Iterating the elements in list 
for val in valList: 
    for ele in range(int(val), int(val) + 2):  
        myDict.setdefault(ele, []).append(val) 

print(myDict)  
print(myDict.keys()) 
print(myDict.values()) 


#%%

# Method #4: Using list comprehension
#-------------------------------------------

# Creating a dictionary of lists using list comprehension 
d = dict((val, range(int(val), int(val) + 2)) for val in ['1', '2', '3']) 
                  
print(d) 

#%%
# Method #5: Using defaultdict
#------------------------------------
#Note that the same thing can also be done with simple dictionary
# but using defaultdict is more efficient for such cases.

# Importing defaultdict 
from collections import defaultdict 
  
lst = [('Geeks', 1), ('For', 2), ('Geeks', 3)] 
orDict = defaultdict(list) 
  
print(orDict)
#%%
# iterating over list of tuples 
for key, val in lst: 
    orDict[key].append(val) 
  
print(orDict) 

#Note that there are only two key:value pairs in output dictionary 
#but the input list contains three tuples.
#The first element(i.e. key) is same for first and third tuple 
#and two keys can never be same.

#%%
# Method #6: Using Json
#------------------------
import json 
  
#Initialisation of list 
lst = [('Geeks', 1), ('For', 2), ('Geeks', 3)] 
  
#Initialisation of dictionary 
dict = {} 
  
#using json.dump() 
hash = json.dumps(lst) 
  
#creating a hash 
dict[hash] = "converted"
  
#Printing dictionary 
print(dict) 
 
#%%














        
        