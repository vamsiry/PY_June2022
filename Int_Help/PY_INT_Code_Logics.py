# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 21:57:31 2022

@author: rvamsikrishna
"""
melt
shift

#File-Contents
#--------------
# 1. Code to print the alphabet and the number of times it appears in a given string
# 2. Merging 2 data frames by cleaning month col in diff styles for month value
# 3. data frame groupby column has list of lists
# 4. Merging morethan 2 data frames in one code

#%%
#%%
#1. Code to print the alphabet and the number of times it appears in a given string
#---------------------------------------------------------------------------------
nstr = "Code to print the alphabet and the number of times it appears in a given string"

#%%
nstr.count("h")

#%%
xx = nstr.strip()
#%%
xxx = pd.Series((list(xx)))

#%%
xxx.value_counts()

#%%
unq = pd.unique(xxx)

#%%
d = {}

for i in unq:
    if i in nstr:
        d[i] += 1
    else:
        d[i] = 1
        
print(d)        
    
#%%
def prCharWithFreq(str):
	d = {}
	for i in str:
		if i in d:
			d[i] += 1
		else:
			d[i] = 1
	for i in str:
		if d[i] != 0:
			print("{}{}{}".format(i,"-",d[i]), end =" ")
			d[i] = 0
	
	
# Driver Code
str = "geeksforgeeks"
prCharWithFreq(str)

#%%
#%%
#2. Merging 2 data frames by cleaning month col in diff styles for month value
#----------------------------------------------------------------------
# import pandas package as pd
import pandas as pd

#Define a dictionary containing students data
data1 = {'Month': ['Jan', 'January', '01', 'Feb',"02","2","Feburary","Mar",
                  "mar","03"],
				'Sales': [222, 129, 250, 178,545,323,7676,345,887,544]}

# Convert the dictionary into DataFrame
df1 = pd.DataFrame(data, columns = ['Month', 'Sales'])

#print("Given Dataframe :\n", Ndf)

data2 = {'Month': ['Jan', 'Jan', 'Jan', 'Feb',"Feb","Feb","Feb","Feb",
                  "Feb","Feb"],
				'Sales': [454, 454, 565, 343,764,323,765,343,887,988]}

df2 = pd.DataFrame(data2, columns = ['Month', 'Sales'])

#%%
df1
#%%
df1.Month.unique()

#%%
xx = {'Jan':'Jan', 'January':'Jan', '01':'Jan', 'Feb':'Feb', '02':'Feb', 
      '2':'Feb', 'Feburary':'Feb', 
      'Mar':'Mar', 'mar':'Mar','03':'Mar'}

#%%
#df['GenderMap'] = df.Gender.map({'Male':1,'Female':0})

df1['Month'] = df1.Month.map(xx)
#%%
df3 = pd.DataFrame(df1.groupby(['Month'])['Sales'].sum(),columns = ['Month', 'Sales'])

#%%
df3.drop("Month",axis=1,inplace=True) #dropping a col in python 

#%%
df3.reset_index()
#%%
left_join = df3.merge(right=df1, how='left', on='Month')
left_join


#%%
#%%
# 3. data frame groupby column has list of lists
#-----------------------------------------------------
dataa = {'A': ['1', '1', '1', '2',"3"],
         'B':[[1,2,3],[2,4],[2,8],[6],[7]],
         'C': ['foo','dss','wew','sss','wew']}

DF = pd.DataFrame(dataa, columns = ['A', 'B','C'])

DF
#%%
DF.groupby('A').agg({'B': 'sum', 'C': lambda x: ' '.join(x)})

#%%
DF.groupby(['A']).sum()

#%%
#%%
## 4. Merging morethan 2 data frames in one code
#------------------------------------------------
#Join Multiple data frames
import pandas as pd
from functools import reduce

# compile the list of dataframes you want to merge
data_frames = [df1, df2, df3]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['key_col'],
                                            how='outer'), data_frames)

#%%

X = [[1,2,3],  
       [4,5,6],  
       [7,8,9]]  
 
Y = [[10,11,12],  
      [13,14,15],  
      [16,17,18]]  
 
result = [[0,0,0],  
               [0,0,0],  
              [0,0,0]]  
 
# iterate through rows of X  
for i in range(len(X)):  
   for j in range(len(Y[0])):  
       for k in range(len(Y)):  
           result[i][j] += X[i][k] * Y[k][j]  
for r in result:  
   print(r)
    
#%%    
import numpy as np
B = np.arange(16).reshape(4, 4)
print(type(B))
print(B)
#%%
A = np.array([[2, 4], [5, -6]])
print(type(A))

#%%
data1 = {'A1': [1,2,3,4,5,6,7,8,9],
				'A2': [222, 129, 250, 178,545,323,7676,345,887]}

# Convert the dictionary into DataFrame
df1 = pd.DataFrame(data1, columns = ['A1', 'A2'])

#%%

import pandas as pd

import numpy as np

#%%
np.sum(df1[['A1','A2']],axis = 1)

#%%

data1 = {'A1': [1,1,1,2,2,3,3,3,3],
				'A2': ['a', 'b', 'a', 'f','f','e','r','r','r']}

# Convert the dictionary into DataFrame
df1 = pd.DataFrame(data1, columns = ['A1', 'A2'])

#%%

df1.groupby(['A1']).(['A2']).count()


#df.groupby(['marital','response'])['age'].mean()























