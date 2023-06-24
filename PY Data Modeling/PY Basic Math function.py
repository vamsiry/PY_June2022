# -*- coding: utf-8 -*-
"""
Created on Sat May 14 21:55:19 2022

@author: rvamsikrishna
"""


test_yourself = [1, 1, 2, 2, 3, 3, 3, 3, 4, 5, 5]

from functools import reduce
  
def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)

Average(test_yourself)

#%%
from statistics import mean
mean(test_yourself)

#%%
test_yourself = [1, 1, 2, 2, 3, 3, 3, 3, 4, 5, 5]

sum(test_yourself)/len(test_yourself)

#%%
test_yourself[round(len(test_yourself) / 2) - 1]    

#%%
#%%
#Print 2 lists side by side
#------------------------------

a = ['a', 'b', 'c']
b = ['1', '0', '0']
res = "\n".join("{} {}".format(x, y) for x, y in zip(a, b))
print(res)

#The zip() function will iterate tuples with the corresponding elements
# from each of the lists

#%%
for i in range(len(a)):
    print(a[i] + '\t ' + b[i])
#%%
for x, y in zip(a, b):
    print(x, y, sep='\t\t')
#%%    
[print(x,y) for x,y in zip(a, b)]

#%%    
#%%

























