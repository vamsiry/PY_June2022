#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 18:55:14 2017

@author: vamsi
"""

#%%
import numpy as np 
a = 'hello world' 
print (a)

#%%

a = np.arange(0,60,5) 
a = a.reshape(3,4) 

print ('Original array is:' )
print (a) 
print ('\n'  )

print ('Sorted in C-style order:' )
for x in np.nditer(a, order = 'C'): 
   print (x),  
print ('\n' )

print ('Sorted in F-style order:' )
for x in np.nditer(a, order = 'F'): 
   print (x),
   
   
#%%
import numpy as np
a = np.arange(0,60,5)
a = a.reshape(3,4)
print ('Original array is:')
print (a)
print ('\n')

for x in np.nditer(a, op_flags=['readwrite']):
 x[...]=2*x
print('Modified array is:')
print(a)   
 
#%%
my_list2 = [[4,5,6,7], [3,4,5,6]]
print(my_list2)

#%%


import numpy as np 
a = np.array([[10,10], [2,3], [4,5]]) 

print ('Our array is:' )
print(a)  
#%%
print ('Create a slice:')
s = a[:, :1] 
print (s) 

#%%
import numpy as np 
from matplotlib import pyplot as plt 

x = np.arange(1,11) 
y = 2 * x + 5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y,"dr") 
plt.show() 
 
