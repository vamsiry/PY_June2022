# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:21:16 2018

@author: vamsi
"""

import numpy as np
#%%
x = np.random.random((5, 20))
print(x)
#%%
x = np.random.random((3, 3, 3))
print(x)

#%%
x = np.random.random((2, 3, 3, 3))
print(x)
print(x.ndim)

#%%
x = np.random.random((3, 3, 3, 3))
print(x)

#%%
np.random.randint(10, size=(100, 1))

#%%
np.random.randint(2, size=(5, 1))

#%%
import numpy as np

height = [10,20,30]
weight = [100,200,300]

type(height)

# vectorized operations not supported in list
height + weight # concatenates vectors instead of doing pair-wise addition
height ** 2 # power of 2 (Squared) not supported
height[height > 20]

# transform list to array
np_height = np.array(height)
np_height
np_weight = np.array(weight)
np_weight
np_weight.shape

np_height[0]
np_height[-1] # 1st element in reverse direction
np_height[0:2] # sliced array
np_height[0:2:2] # jump by 2 positions; index 2-1=1 is crossed after 1st jump itself

# vectorized operations supported on 'numpy' arrays
np_height + np_weight
np_height ** 2   # squaring values
np_height > 20
np_height[np_height > 15]

# Dot product of arrays/vectors (a,b,c).dot(d,e,f) = ad + be + cf
# Needed as a part of matrix multiplications
np_height.dot(np_weight)

# statistical operations on vectors
np_height.min()
np_height.max()
np_height.argmin() # index of the minimum
np_height.sort()

# ctrl+i for help

# Creating arrays of 0s and 1s
zero1 = np.zeros(2) # floating type 0s by default
zero1.shape
zero1
type(zero1)
zero1.shape


# Python uses tuples for argument passing, since tuples are immutable
zero2 = np.zeros(2, int) # int type 0s
zero2.shape
zero2

ones1 = np.ones(3)
ones1
ones2 = np.ones(3, int)
ones2

# list is heterogenous; array is homogenous
list1 = [10, True, 'abc']
type(list1)
list1
# casting heterogenous list to array; everything is cast to a string
array1 = np.array(list1)
array1
array1 + array1 # not supported since type is 'S11'. 'str' types would have been concatenated

np.mean(height)
np.mean(np_height)

a1 = np.array([[1,2,3], [4,5,6]])
a1.shape
type(a1)
a1[1,1]
a1[1,] # 2nd row displayed
a1[:,1] # Error; not supported

#%%
import numpy as np

a1 = np.array([[1,2,3], [4,5,6]])
a1.shape
a1[1,1]

# use colon to get values across dimensions
a1[1,:] # 2nd row, all columns
a1[:,1] # 2nd column, all rows

# colon also used for range
a1[0:2,1] # 1st and 2nd rows, 2nd column

# create zero matrix of size 2 x 3
a2 = np.zeros((2,3), int)
a2.shape
a2

# create ones matrix of size 3 by 2
a3 = np.ones((3,2), int)
a3.shape
a3

# create identity matrix of size 3 x 3
a4 = np.eye(3,3,dtype=int)
a4.shape
a4

# reshape the matrix
a5 = np.array([[1,2],[3,4],[5,6]])
a5
type(a5)
a5.shape

a5.reshape(2,3)

tmp = a5.reshape((1,6)) # 2D array only; not a 1D array
type(tmp)

# reshape a matrix into a single row
a6 = a5.reshape((1,-1))
# if we don't remember how many elements are present; "-1" means all elements
a6
type(a6)
a6.shape


# get back the original original
a7 = a6.reshape(3,2)
a7

# getting useful statistics on matrix
a1
# 0 -> column statistics; 1 -> row statistics
a1.max(axis = 0) # max value from each column
a1.max(axis = 1) # max value from each row
a1.mean(axis = 1)
a1.std(axis = 0)

# element-wise operations on matrices
a7 = np.array([[1,2],[3,4]])
a8 = np.array([[1,1],[2,2]])
a7 + a8
a7 * a8 # element-wise multiplication

# matrix multiplication
a7.dot(a8) # dot product

# matrix transpose
a7.T
#%%
# Source : 






























