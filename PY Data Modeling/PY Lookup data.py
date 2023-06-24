# -*- coding: utf-8 -*-
"""
Created on Tue May 17 19:11:54 2022

@author: rvamsikrishna
"""
#%%
#Contents
#============
#Data lookup
#merge
#join
#concat


#%%
# Data lookup

import pandas as pd
 
df = pd.read_csv("C://Users//rvamsikrishna//Desktop//PY//Python//PY Data Modeling//mtcars.csv")
df.index

df_Target = pd.DataFrame({"model":['Duster 360','Ferrari Dino','Honda Civic','Lotus Europa','Volvo 142E']})
df_Target

#Create a dataframe for to store subset
df_Target['kmpl'] = ""
df_Target['cyl'] = ""
df_Target['hp'] = ""
df_Target

#Set the index
df_Target.set_index('model',inplace= True)
df_Target

#Data mapping
df_Target["kmpl"] = df_Target.index.map(df["mpg"]) * 0.4
df_Target["hp"] = df_Target.index.map(df["hp"])
df_Target["cyl"] = df_Target.index.map(df["cyl"])
df_Target

#df1['sex'] = df1.Name.map(df2.set_index('Player')['Gender'])
#or
#d = df2.set_index('Player')['Gender'].to_dict()
#df1['sex'] = df1.Name.map(d)



#%%
#%%
#%%

#Python Pandas DataFrame Join, Merge, and Concatenate
#====================================================

#Merge : merge() is used to combine two (or more) dataframes on the basis of 
#values of common columns (indices can also be used, use left_index=True and/or 
#right_index=True), 

#Join : join() is used to merge 2 dataframes on the basis of the index or key column:
# instead of using merge() with the option left_index=True we can use join().


#Concat : concat() is used to append one (or more) dataframes one below the 
#other (or sideways, depending on whether the axis option is set to 0 or 1).


#%%

#Join-- . It will combine all the columns from the two tables, with the 
#***********
# common columns renamed with the defined lsuffix and rsuffix. The way that rows
# from the two tables are combined is defined by how.

#how='left'/right/inner/outer
df = pd.join(df1, df2, on=None, how='left', lsuffix='', rsuffix='', sort=False)


#%%
#Merge-- 
#*********

#.merge() first aligns two DataFrame' selected common column(s) or index, and
# then pick up the remaining columns from the aligned rows of each DataFrame.

#Similar to join, merge also combines all the columns from the two tables,
# with the common columns renamed with the defined suffixes

#However, merge provides three ways of flexible control over row-wise alignment.

#The first way is to use on = COLUMN NAME, here the given column must be the 
#common column in both tables.  

#The second way is to use left_on = COLUMN NAME and right_on = COLUMN NAME , 
#and it allows to align the two tables using two different columns. 

#The third way is to use left_index = True and right_index = True, and the two 
#tables are aligned based on their index.


df = pd.merge(df1, df2, how='inner', on=None, left_on=None, right_on=None,
              left_index=False, right_index=False, 
              sort=False, suffixes=('_x', '_y'), copy=True,
              indicator=False, validate=None)


#Method_2 Using Joining functions
# read csv data
df1 = pd.read_csv('Student_data.csv')
df2 = pd.read_csv('Course_enrolled.csv')
   
#how='left'/right/inner/outer
inner_join = pd.merge(df1, df2, on ='Col_Name', how ='inner')

#Merge only subset of columns
df = pd.merge(df,df2[['Key_Column','Target_Column']],on='Col_Name', how='left')


new = pd.merge(df1,df2, left_on='Name', right_on='Player').rename(columns={'Gender':'sex'}).drop('Player', axis=1)


new = df1.merge(df2[['a', 'b', 'key1']], how = 'left',left_on = 'key2', right_on = 'key1').drop(columns = ['key1'])

#%%
#Cancatenate---
#**************
#https://stackoverflow.com/questions/38256104/differences-between-merge-and-concat-in-pandas

#.concat() simply stacks multiple DataFrame together either vertically, or 
#stitches horizontally after aligning on index


df = pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, 
                   keys=None, levels=None, names=None, verify_integrity=False, 
                   sort=None, copy=True)


df2=pd.concat([df]*2, ignore_index=True) #double the rows of a dataframe

df2=pd.concat([df, df.iloc[[0]]]) # add first row to the end

df3=pd.concat([df1,df2], join='inner', ignore_index=True) # concat two df's

#%%
    




