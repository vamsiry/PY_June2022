# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:57:13 2022

@author: vamsi
"""

#https://www.w3schools.com/python/numpy/numpy_data_types.asp

#https://numpy.org/doc/stable/user/quickstart.html

#NumPy is short for "Numerical Python"


#NumPy is a Python library and is written partially in Python, but most of
# the parts that require fast computation are written in C or C++.


#NumPy is used for working with arrays.
#NumPy’s object is the homogeneous multidimensional array. 
#It is a table of elements (usually numbers), all of them are same type,
# indexed by a tuple of non-negative integers.
#In NumPy dimensions are called axes.

#For example, the array for the coordinates of a point in 3D space,
# [1, 2, 1], has one axis. That axis has 3 elements in it, so we say 
#it has a length of 3.

#In the Example below, the array has 2 axes. The first axis has 
#a length of 2, the second axis has a length of 3.

#[[1., 0., 0.],[0., 1., 2.]]




#%%
#Operations using NumPy
#-------------------------

#Using NumPy, a developer can perform the following operations −

#Mathematical and logical operations on arrays.

#Fourier transforms and reshaping the data stored in multidimensional arrays.

#NumPy has in-built functions for linear algebra,matrices and random number generation.



#%%
#Why Use NumPy?

#NumPy array object that is up to 50x faster than traditional Python lists.

#The array object in NumPy is called ndarray, it provides a lot of 
#supporting functions that make working with ndarray very easy.


#%%

#Checking NumPy Version

import numpy as np

print(np.__version__)

#%%
#np.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)

#To create an ndarray, we can pass a list, tuple or any array-like object 
#into the array() method, and it will be converted into an ndarray:
    
    
#%%
#Creating array using list
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))
print(arr.dtype)

#%%
#Creating array using Tuple
arr = np.array((1, 2, 3, 4, 5))
print(arr) 
print(type(arr))

#%%
a = np.array(42) #0-D Arrays

b = np.array([1, 2, 3, 4, 5]) #1-D Arrays

c = np.array([[1, 2, 3], [4, 5, 6]]) #2-D Arrays

d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]) #3-D arrays

print(a)
print(b)
print(c)
print(d)

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim) 

#%%
#An array can have any number of dimensions.

#When the array is created, you can define the number of dimensions
# by using the ndmin argument.

#%%
#create array with 5 dimensions using ndmin using a vector with values 1,2,3,4
arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('number of dimensions :', arr.ndim) 
print('shape of array :', arr.shape)
print('shape of array :', arr.size)


#%%
#The more important attributes of an ndarray object are:
a = np.arange(15).reshape(3, 5)
print(a)

print(type(a))

print(a.ndim) #number of dimensions of an array


print(a.shape) #returns (3, 5), which means that the array has 2 dimensions, 
#where the first dimension has 3 elements and the second has 5.


print(a.size) # total number of elements of the array.
# This is equal to the product of the elements of shape.


print(a.dtype) #Data Type of an Array

print(a.astype('f')) #Converting Data Type to float ('bool')

print(a.itemsize) #size in bytes of each element of the array.

#%%
#Accessing Elements of an Array 
#--------------------------------
arr = np.array([1, 2, 3, 4])

print(arr[0]) 
print(arr[1])

print(arr[2] + arr[3]) 

#%%
#Accessing Elements of an 2D Array 
#--------------------------------

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print(arr)

print('2nd element on 1st row: ', arr[0, 1]) 

print('5th element on 2nd row: ', arr[1, 4]) 

#%%
#Accessing Elements of an 3D Array 
#--------------------------------
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr)
print(arr[0, 1, 2]) 

#%%
#Negative Indexing
#-------------------
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('Last element from 2nd dim: ', arr[1, -1]) 

#%%
#Slicing arrays : [start:end:step]
#--------------------------------------
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5]) 
print(arr[4:])
print(arr[:4]) 
print(arr[1:5:2])
print(arr[::2]) 

#Negative Slicing
print(arr[-3:-1]) #Slice from the index 3 from the end to index 1 from the end:

#%%
#Slicing 2-D Arrays
#---------------------------
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[1, 1:4]) 

print(arr[0:2, 2]) #From both elements, return index 2:
    
print(arr[0:2, 1:4]) #From both elements, slice index 1 to index 4 
#(not included), this will return a 2-D array:
    
#%%
#NumPy Array Copy vs View
#--------------------------------
#copy is a new array,and the view is just a view of the original array.

arr = np.array([1, 2, 3, 4, 5])

x = arr.copy()
y = arr.view()

print(x.base)
print(y.base)

#%%
#Reshaping arrays
#-----------------------
#Reshaping means changing the shape of an array.

#The shape of an array is the number of elements in each dimension.

#By reshaping we can add or remove dimensions or change number of 
#elements in each dimension.

#Reshape From 1-D to 2-D
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(4, 3)

print(newarr) 

#%%
#Reshape From 1-D to 3-D
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(2, 3, 2)

print(newarr) 

#%%
#Check if the returned array is a copy or a view:
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

print(arr.reshape(2, 4).base) 

#The example above returns the original array, so it is a view.

#%%
#Unknown Dimension
#you do not have to specify an exact number for one of the dimensions
#in the reshape method
#Pass -1 as the value, and NumPy will calculate this number for you.
#Note: We can not pass -1 to more than one dimension.
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
newarr = arr.reshape(2, 2, -1)
print(newarr)

#%%
#Flattening the arrays
#Flattening array means converting a multidimensional array into a 1D array.
#We can use reshape(-1) to do this.
arr = np.array([[1, 2, 3], [4, 5, 6]])
newarr = arr.reshape(-1)
print(newarr) 

#Note: There are a lot of functions for changing the shapes of arrays in 
#numpy flatten, ravel and also for rearranging the elements rot90, flip,fliplr,
# flipud etc. These fall under Intermediate to Advanced section of numpy.

#%%
#Iterating Arrays
#-------------------------
#Iterating means going through elements one by one.

#As we deal with multi-dimensional arrays in numpy, we can do this using 
#basic for loop of python.

#Iterating 1-D Arrays
arr = np.array([1, 2, 3])

for x in arr:
  print(x) 
  
#%%  
#Iterating 2-D Arrays
arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  print(x) 
  
#%%  
#If we iterate on a n-D array it will go through n-1th dimension one by one.

#To return the actual values, the scalars, we have to iterate the arrays
# in each dimension.

arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  for y in x:
    print(y) 
    
#%%   
#Iterating 3-D Arrays
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  print(x) 
  
#%%
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  for y in x:
    for z in y:
      print(z) 
      
#%%      
#Iterating Arrays Using nditer()

#nditer() is a helping function that can be used from very basic
#to very advanced iterations

#In basic for loops, iterating through each scalar of an array we need 
#to use n for loops which can be difficult to write for arrays with very
# high dimensionality.

arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

#arr = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]])

print(arr.ndim)
print(arr.shape)

for x in np.nditer(arr):
  print(x) 
  
#%%  
#Iterating Array With Different Data Types

#We can use op_dtypes argument and pass it the expected datatype to
# change the datatype of elements while iterating.

#NumPy does not change the data type of the element in-place so it needs 
#some other space to perform this action, that extra space is called buffer,
# and in order to enable it in nditer() we pass flags=['buffered'].

arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x) 

#%%
#Iterating With Different Step Size

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for x in np.nditer(arr[:, ::2]):
  print(x) 
  
#%%  
# Enumerated Iteration Using ndenumerate()

#Enumeration means mentioning sequence number of somethings one by one.

#Sometimes we require corresponding index of the element while iterating, 
#the ndenumerate() method can be used for those usecases.

arr = np.array([1, 2, 3])

for idx, x in np.ndenumerate(arr):
  print(idx, x) 
  
#%%  
#Enumerate on following 2D array's elements:
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for idx, x in np.ndenumerate(arr):
  print(idx, x)     

#%%
#Joining NumPy Arrays
#----------------------
#Joining means putting contents of two or more arrays in a single array.

#In SQL we join tables based on a key, whereas in NumPy we join arrays by axes.

#We pass a sequence of arrays that we want to join to the concatenate() 
#function, along with the axis. If axis is not explicitly passed, 
#it is taken as 0.

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr) 

#%%
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

arr = np.concatenate((arr1, arr2),axis = 1)
print(arr) 

#%%
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

arr = np.concatenate((arr1, arr2), axis = 0)
print(arr) 

print(arr1.ndim)
print(arr.ndim)

print(arr1.shape)
print(arr.shape)

#%%
#Join two 2-D arrays along rows (axis=1):
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

arr = np.concatenate((arr1, arr2), axis=1)
print(arr)    

print(arr1.ndim)
print(arr.ndim)

print(arr1.shape)
print(arr.shape)

#%%
#Joining Arrays Using Stack Functions

#Stacking is same as concatenation, the only difference is that
# stacking is done along a new axis.

#We can concatenate two 1-D arrays along the second axis which would 
#result in putting them one over the other, ie. stacking.

#We pass a sequence of arrays that we want to join to the stack() method
# along with the axis. If axis is not explicitly passed it is taken as 0.

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)
print(arr) 

print(arr1.ndim)
print(arr.ndim)

print(arr1.shape)
print(arr.shape)


#%%
#Stacking Along Rows--hstack() to stack along rows

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

arr = np.hstack((arr1, arr2))
print(arr) 

print(arr1.ndim)
print(arr.ndim)

print(arr1.shape)
print(arr.shape)

#%%
#Stacking Along Columns--vstack()  to stack along columns.

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

arr = np.vstack((arr1, arr2))
print(arr) 

print(arr1.ndim)
print(arr.ndim)

print(arr1.shape)
print(arr.shape)

#%%
#Stacking Along Height (depth)-dstack() to stack along height,
# which is the same as depth.

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

arr = np.dstack((arr1, arr2))
print(arr)

print(arr1.ndim)
print(arr.ndim)

print(arr1.shape)
print(arr.shape)

#%%
#Splitting NumPy Arrays
#-----------------------------

#Joining merges multiple arrays into one and Splitting breaks one 
#array into multiple.

#We use array_split() for splitting arrays, we pass it the array we want
# to split and the number of splits.

arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)
print(newarr)

#%%
#Note: The return value is an array containing three arrays.
#If the array has less elements than required, it will adjust from 
#the end accordingly.
arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 4)
print(newarr)

#Note: We also have the method split() available but it will not adjust 
#the elements when elements are less in source array for splitting 
#like in example above, array_split() worked properly but split() would fail.

#%%
#Split Into Arrays
#The return value of the array_split() method is an array 
#containing each of the split as an array.

#If you split an array into 3 arrays, you can access them from the 
#result just like any array element:
    
arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)

print(newarr[0])
print(newarr[1])
print(newarr[2]) 

#%%
#Splitting 2-D Arrays

arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

newarr = np.array_split(arr, 3)
print(newarr)

print(newarr[0])
print(newarr[1])
print(newarr[2]) 

#%%
#The example above returns three 2-D arrays..

#In addition, you can specify which axis you want to do the split around.

#The example below also returns three 2-D arrays, but they are 
#split along the row (axis=1).

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.array_split(arr, 3, axis=1)
print(newarr) 

#%%
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

newarr = np.array_split(arr, 3,axis=1)
print(newarr)

print(type(newarr[0]))
print(newarr[0].ndim)
print(newarr[0].shape)

#%%
#An alternate solution is using hsplit() opposite of hstack()
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.hsplit(arr, 3)
print(newarr)

#Note: Similar alternates to vstack() and dstack() are available
# as vsplit() and dsplit().
#%%
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.vsplit(arr, 3)
print(newarr)


#%%
#Searching Arrays
#-----------------------
#You can search an array for a certain value, and return the 
#indexes that get a match.

#To search an array, use the where() method.

arr = np.array([1, 2, 3, 4, 5, 4, 4])

x = np.where(arr == 4)
print(x) 

#%%
# Find the indexes where the values are even:
  
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

x = np.where(arr%2 == 0)
print(x) 
print(arr[np.where(arr%2 == 0)])

#%%
#Find the indexes where the values are odd:

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

x = np.where(arr%2 == 1)
print(x)

#%%

#Search Sorted--There is a method called searchsorted() which performs a 
#binary search in the array, and returns the index where the specified 
#value would be inserted to maintain the search order.

#The searchsorted() method is assumed to be used on sorted arrays.

#Find the indexes where the value 7 should be inserted:
arr = np.array([6, 7, 8, 9])

x = np.searchsorted(arr, 7)
print(x) 

#Example explained: The number 7 should be inserted on index 1 to 
#remain the sort order.

#The method starts the search from the left and returns the first index
# where the number 7 is no longer larger than the next value.

#%%
#Search From the Right Side
arr = np.array([6, 7, 8, 9])

x = np.searchsorted(arr, 7, side='right')
print(x) 

#%%
#Multiple Values--Find the indexes where the values 2, 4, and 6 should be inserted:
arr = np.array([1, 3, 5, 7])

x = np.searchsorted(arr, [2, 4, 6])
print(x)     
    
#%%
#Filtering Arrays
#-----------------------
#Getting some elements out of an existing array and creating a 
#new array out of them is called filtering.

#In NumPy, you filter an array using a boolean index list.
arr = np.array([41, 42, 43, 44])

x = [True, False, True, False]
newarr = arr[x]
print(newarr) 

#%%
#Create a filter array that will return only values higher than 42:
arr = np.array([41, 42, 43, 44])

# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr:
  # if the element is higher than 42, set the value to True, otherwise False:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)     
    
#%%
#Create a filter array that will return only even elements 
#from the original array:

arr = np.array([1, 2, 3, 4, 5, 6, 7])

# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr:
  # if the element is completely divisble by 2, set the value to True, otherwise False
  if element % 2 == 0:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)
print(newarr) 

#%%    
arr = np.array([1, 2, 3, 4, 5, 6, 7])

# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr:
    if element %2 ==0:
        filter_arr.append(element)
    
print(filter_arr)    

#%%
#Creating Filter Directly From Array

#Create a filter array that will return only values higher than 42:
arr = np.array([41, 42, 43, 44])

filter_arr = arr > 42
newarr = arr[filter_arr]

print(filter_arr)
print(newarr)     

#%%
arr = np.array([1, 2, 3, 4, 5, 6, 7])

filter_arr = arr % 2 == 0
newarr = arr[filter_arr]

print(filter_arr)
print(newarr) 

#%%














#%%

print(np.zeros((3, 4))) #2D #creates an array full of zeros,

print(np.ones((2, 3, 4), dtype=np.int16) ) #3D #creates an array full of ones

print(np.empty((2, 3)) ) #creates an array whose initial content is random and depends on the state of the memory.

#%%

#To create sequences of numbers, NumPy provides the arange function
# which is analogous to the Python built-in range, but returns an array.

print(np.arange(10, 30, 5))

print(np.arange(0, 2, 0.3))  # it accepts float arguments

#%%
#












































