#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 00:02:25 2017

@author: vreddy
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf

os.chdir("/home/vreddy/SharedFolder/Practice/data sets/FKD/")
# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

df1 = pd.read_csv("training.csv")

type(df1) # to the object type(like list or data frame)

df1.count() #to the the coutnts for each variable

df1.shape #to get the dimensions

df1.dropna().shape #to see no.of samples have non missing values

df1.columns #to get the column names

df1.head()
df1.tail()

df1.describe()


df1["Image"]
image1=pd.DataFrame(df1['Image'].str.split(' ').tolist())

X_train = np.array(image1).astype(np.int32)
X_train.shape[2,]

plt.matshow(np.reshape(X_train[6,],[96,96]))
plt.imshow(np.reshape(X_train[7,],[96,96]))
plt.show()

#%%
#renaming column names


#The rename method can take a function, for example:

In [11]: df.columns
Out[11]: Index([u'$a', u'$b', u'$c', u'$d', u'$e'], dtype=object)

In [12]: df.rename(columns=lambda x: x[1:], inplace=True)

In [13]: df.columns
Out[13]: Index([u'a', u'b', u'c', u'd', u'e'], dtype=object)

#---------or------------------
df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
# Or rename the existing DataFrame (rather than creating a copy) 
df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)

#----------------or----------------
df = pd.DataFrame({'$a':[1,2], '$b': [10,20]})
df

df.columns = ['a', 'b']
df
#-------------------or----------------
df.rename(columns=lambda x: x.lstrip(), inplace=True)

# I needed to remove more than just whitespace (stripe), so : 
#t.columns = t.columns.str.replace(r'[^\x00-\x7F]+','')

df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True) 

df.Column_1_Name #instead of writing    

df.loc[:, 'Column 1 Name']

#%%

a = [1, 2, 3]
b = [4, 5, 6]

[(x, y) for x in a for y in b]


#%%
# cross join 
dfA = pd.DataFrame({'A':list('ab'), 'B':range(2,0,-1)})
dfB = pd.DataFrame({'C':range(2), 'D':range(4, 6)})
pd.concat([dfA, dfB])

dfA["key"] = 1; dfB["key"] = 1;
m = dfA.merge(dfB,how='outer'); del m["key"];
m


#======================================================================================

#%%
import os
import pandas as pd
import numpy as np
import tensorflow as tf

os.chdir("/home/vreddy/SharedFolder/Practice/data sets/FKD/")
# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

df1 = pd.read_csv("training.csv")

type(df1) # to the object type(like list or data frame)

df1.count() #to the the coutnts for each variable

df1.shape #to get the dimensions

df1.dropna().shape #to see no.of samples have non missing values

df1.columns #to get the column names

df1.head()
df1.tail()

df1.describe()


df1["Image"]
image1=pd.DataFrame(df1['Image'].str.split(' ').tolist())

X_train = np.array(image1).astype(np.int32)
X_train.shape[2,]

plt.matshow(np.reshape(X_train[6,],[96,96]))
plt.imshow(np.reshape(X_train[7,],[96,96]))
plt.show()


#%%













