# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:44:01 2022

@author: rvamsikrishna
"""
#File-Contents
#--------------
#Difference between apply, map and applymap methods in Pandas

#First major difference: DEFINITION
#-------------------------------------
#map is defined on Series ONLY
#applymap is defined on DataFrames ONLY
#apply is defined on BOTH
    
#Second major difference: INPUT ARGUMENT
#-----------------------------------------
# map accepts dicts, Series, or callable
# applymap and apply accept callables only

#Third major difference: BEHAVIOR
#---------------------------------
#map is elementwise for Series
#applymap is elementwise for DataFrames
#apply also works elementwise but is suited to more complex operations 
#and aggregation. The behaviour and return value depends on the function.

#Fourth major difference (the most important one): USE CASE
#-------------------------------------------------------------
#map is meant for mapping values from one domain to another, 
#so is optimised for performance (e.g., df['A'].map({1:'a', 2:'b', 3:'c'}))

#applymap is good for elementwise transformations across
#multiple rows/columns (e.g., df[['A', 'B', 'C']].applymap(str.strip))

#apply is for applying any function that cannot be vectorised 
#(e.g., df['sentences'].apply(nltk.sent_tokenize)).

#%%
gfg_string = 'geeksforgeeks'
xx = 5 * [pd.Series(list(gfg_string))]
#print(xx)
print(xx[0:8])




















