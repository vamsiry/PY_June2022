# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:59:11 2019

@author: rvamsikrishna
"""
#https://www.programiz.com/python-programming/examples/add-matrix

#Python Matrices and NumPy Arrays
#====================================

#A matrix is a two-dimensional data structure where numbers are arranged into rows and columns.

#Python doesn't have a built-in type for matrices. However, we can treat
# list of a list as a matrix. For example:
A = [[1, 4, 5], 
    [-5, 8, 9]]

#Let's see how to work with a nested list.
A = [[1, 4, 5, 12], 
    [-5, 8, 9, 0],
    [-6, 7, 11, 19]]
print("A =", A) 
print("A[1] =", A[1])      # 2nd row
print("A[1][2] =", A[1][2])   # 3rd element of 2nd row
print("A[0][-1] =", A[0][-1])   # Last element of 1st Row
column = [];        # empty list
for row in A:
  column.append(row[2])   
print("3rd column =", column)

#%%

#Matrix Addition using Nested Loop
# Program to add two matrices using nested loop

X = [[12,7,3],
    [4 ,5,6],
    [7 ,8,9]]

Y = [[5,8,1],
    [6,7,3],
    [4,5,9]]

result = [[0,0,0],
         [0,0,0],
         [0,0,0]]

# iterate through rows
for i in range(len(X)):
   # iterate through columns
   for j in range(len(X[0])):
       result[i][j] = X[i][j] + Y[i][j]

for r in result:
   print(r)
   
#%%
#Matrix Addition using Nested List Comprehension
# Program to add two matrices
# using list comprehension

X = [[12,7,3],
    [4 ,5,6],
    [7 ,8,9]]

Y = [[5,8,1],
    [6,7,3],
    [4,5,9]]

result = [[X[i][j] + Y[i][j]  for j in range(len(X[0]))] for i in range(len(X))]

for r in result:
   print(r)

#%%
#Matrix Transpose using Nested Loop
X = [[12,7],
    [4 ,5],
    [3 ,8]]

result = [[0,0,0],
         [0,0,0]]

# iterate through rows
for i in range(len(X)):
   # iterate through columns
   for j in range(len(X[0])):
       result[j][i] = X[i][j]

for r in result:
   print(r)

#%%   
#Matrix Transpose using Nested List Comprehension

''' Program to transpose a matrix using list comprehension'''

X = [[12,7],
    [4 ,5],
    [3 ,8]]

result = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]

for r in result:
   print(r)

#%%
#Matrix Multiplication using Nested Loop

# Program to multiply two matrices using nested loops

# 3x3 matrix
X = [[12,7,3],
    [4 ,5,6],
    [7 ,8,9]]
# 3x4 matrix
Y = [[5,8,1,2],
    [6,7,3,0],
    [4,5,9,1]]
# result is 3x4
result = [[0,0,0,0],
         [0,0,0,0],
         [0,0,0,0]]

# iterate through rows of X
for i in range(len(X)):
   # iterate through columns of Y
   for j in range(len(Y[0])):
       # iterate through rows of Y
       for k in range(len(Y)):
           result[i][j] += X[i][k] * Y[k][j]

for r in result:
   print(r)

#%%
#Matrix Multiplication Using Nested List Comprehension

# Program to multiply two matrices using list comprehension

# 3x3 matrix
X = [[12,7,3],
    [4 ,5,6],
    [7 ,8,9]]

# 3x4 matrix
Y = [[5,8,1,2],
    [6,7,3,0],
    [4,5,9,1]]

# result is 3x4
result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]

for r in result:
   print(r)
#%%
#====================
#NumPy Array
#=================   

#NumPy is a package for scientific computing which has support for a powerful 
#N-dimensional array object.

#NumPy provides multidimensional array of numbers (which is actually an object)
   
import numpy as np
a = np.array([1, 2, 3])
print(a)               # Output: [1, 2, 3]
print(type(a))         # Output: <class 'numpy.ndarray'>

#As you can see, NumPy's array class is called ndarray.

#%%

#How to create a NumPy array?
#-------------------------------------
#There are several ways to create NumPy arrays.

#1. Array of integers, floats and complex Numbers

import numpy as np
A = np.array([[1, 2, 3], [3, 4, 5]])
print(A)
A = np.array([[1.1, 2, 3], [3, 4, 5]]) # Array of floats
print(A)
A = np.array([[1, 2, 3], [3, 4, 5]], dtype = complex) # Array of complex numbers
print(A)

#%%
#2. Array of zeros and ones
#----------------------------------------------    
import numpy as np
zeors_array = np.zeros( (2, 3) )
print(zeors_array)
'''
 Output:
 [[0. 0. 0.]
  [0. 0. 0.]]
'''
ones_array = np.ones( (1, 5), dtype=np.int32 ) # specifying dtype
print(ones_array)      # Output: [[1 1 1 1 1]]

#%%
#3. Using arange() and shape()
#-------------------------------------
import numpy as np
A = np.arange(4)
print('A =', A)
B = np.arange(12).reshape(2, 6)
print('B =', B)

#%%

#Matrix Operations
#---------------------------
#addition of two matrices, multiplication of two matrices and transpose of a matrix.
# We used nested lists before to write those programs. Let's see how we can do the 
#same task using NumPy array.

#Addition of Two Matrices
#----------------------------
import numpy as np
A = np.array([[2, 4], [5, -6]])
B = np.array([[9, -3], [3, 6]])
C = A + B      # element wise addition
print(C)

#%%
#Multiplication of Two Matrices
#-----------------------------------

#To multiply two matrices, we use dot() method.

#Note: * is used for array multiplication (multiplication of corresponding elements 
#of two arrays) not matrix multiplication.

import numpy as np
A = np.array([[3, 6, 7], [5, -3, 0]])
B = np.array([[1, 1], [2, 1], [3, -3]])
C = a.dot(B)
print(C)

#%%
#Transpose of a Matrix
#--------------------------

#We use numpy.transpose to compute transpose of a matrix.
import numpy as np
A = np.array([[1, 1], [2, 1], [3, -3]])
print(A.transpose())

#%%

#Access matrix elements, rows and columns
#----------------------------------------------

#Similar like lists, we can access matrix elements using index.
# Let's start with a one-dimensional NumPy array.

import numpy as np
A = np.array([2, 4, 6, 8, 10])
print("A[0] =", A[0])     # First element     
print("A[2] =", A[2])     # Third element 
print("A[-1] =", A[-1])   # Last element     

#%%
#Now, let's see how we can access elements of a two-dimensional array 
#(which is basically a matrix).

import numpy as np
A = np.array([[1, 4, 5, 12],
    [-5, 8, 9, 0],
    [-6, 7, 11, 19]])
#  First element of first row
print("A[0][0] =", A[0][0])  
# Third element of second row
print("A[1][2] =", A[1][2])
# Last element of last row
print("A[-1][-1] =", A[-1][-1])     

#%%
#Access rows of a Matrix
#--------------------------------
import numpy as np
A = np.array([[1, 4, 5, 12], 
    [-5, 8, 9, 0],
    [-6, 7, 11, 19]])
print("A[0] =", A[0]) # First Row
print("A[2] =", A[2]) # Third Row
print("A[-1] =", A[-1]) # Last Row (3rd row in this case)

#%%
#Access columns of a Matrix
#----------------------------------

import numpy as np
A = np.array([[1, 4, 5, 12], 
    [-5, 8, 9, 0],
    [-6, 7, 11, 19]])
print("A[:,0] =",A[:,0]) # First Column
print("A[:,3] =", A[:,3]) # Fourth Column
print("A[:,-1] =", A[:,-1]) # Last Column (4th column in this case)

#%%
#Slicing of a Matrix
#------------------------
#Slicing of a one-dimensional NumPy array is similar to a list.

import numpy as np
letters = np.array([1, 3, 5, 7, 9, 7, 5])
# 3rd to 5th elements
print(letters[2:5])        # Output: [5, 7, 9]
# 1st to 4th elements
print(letters[:-5])        # Output: [1, 3]   
# 6th to last elements
print(letters[5:])         # Output:[7, 5]
# 1st to last elements
print(letters[:])          # Output:[1, 3, 5, 7, 9, 7, 5]
# reversing a list
print(letters[::-1])          # Output:[5, 7, 9, 7, 5, 3, 1]

#%%
#Now, let's see how we can slice a matrix.
#-----------------------------------------------

import numpy as np
A = np.array([[1, 4, 5, 12, 14], 
    [-5, 8, 9, 0, 17],
    [-6, 7, 11, 19, 21]])
print(A[:2, :4])  # two rows, four columns
''' Output:
[[ 1  4  5 12]
 [-5  8  9  0]]
'''
print(A[:1,])  # first row, all columns
''' Output:
[[ 1  4  5 12 14]]
'''
print(A[:,2])  # all rows, second column
''' Output:
[ 5  9 11]
'''
print(A[:, 2:5])  # all rows, third to fifth column
'''Output:
[[ 5 12 14]
 [ 9  0 17]
 [11 19 21]]
'''

#%%









































