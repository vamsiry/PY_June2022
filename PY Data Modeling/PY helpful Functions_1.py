# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:38:51 2022

@author: rvamsikrishna
"""
#%%
#%%
#*************************2. zip()***********************
#The zip() function, for parallel iteration, is very handy when we need to
# iterate over multiple lists.
#This function is generally used with a loop and to compare similar indexed 
#elements in each list. 
#Here we can use any number of lists inside the zip() function to iterate over them.
    
a = ("John", "Charles", "Mike")
b = ("Jenny", "Christy", "Monica")
x = zip(a, b)
print(tuple(x))

#%%
a = ["John", "Charles", "Mike"]
b = ["Jenny", "Christy", "Monica"]
for i,j in zip(a,b):
    print(f"{i} will marry {j}")


#%%
#%%
#**********************************5. in******************************

#This is used to check whether an object contains any other object. 
#The object can be a string, list, dictionary, or tuple.

bikes = ['trek', 'redline', 'giant']
ans='trek' in bikes
print(ans)


#%%
#%%
#****************************7. join()*******************************
#This function is mostly used with the split function in Python. 
#While doing the split, we try to split the string into small strings. 
#And on the other hand, the join() function joins all the split items.

myTuple = ("John", "Peter", "Vicky")
x = " ".join(myTuple)
print(x)

    
#%%
#%%
#****************************9. split()**********************************
#This function is mostly used while taking the user input using the input() function.
# It takes the object as a string and splits the object based on anything.

#If you want to split a string with the keyword “f,” you can pass this inside
# the split(“f”) function.

txt = "welcome to the medium"
x = txt.split()
print(x)
#%%
txt = "2 6 9 5 8 14 25 -15"
x = [int(i)*10 for i in txt.split()]
print(x)


#%%


#%%

ac = [1, 1, 2, 2, 3, 3, 3, 3, 4, 5, 5]

def average(x):
    return sum(x)/len(x)

average(ac)

#%%
#

#%%


#%%


























