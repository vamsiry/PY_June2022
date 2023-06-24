# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:46:50 2022

@author: rvamsikrishna
"""

#File-Contents
#--------------
#Merge
#join
#Concat


#%%
import pandas as pd

df = pd.read_csv("C://Users//rvamsikrishna//Desktop//PY//Python//PY Data Modeling//marketing_analysis.csv",skiprows=1, low_memory=False)

#url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/HairEyeColor.csv"
#df = pd.read_csv(url)

#%%
#%%
#Concatenation 
#---------------
df1 = pd.DataFrame({'name': ['John', 'Smith','Paul'],
                     'Age': ['25', '30', '50']},
                    index=[0, 1, 2])
df2 = pd.DataFrame({'name': ['Adam', 'Smith' ],
                     'Age': ['26', '11']},
                    index=[3, 4])  

df_concat = pd.concat([df1,df2]) 
df_concat

#%%
df_concat.drop_duplicates('name',inplace=True) #Drop_duplicates by column
#%%
df_concat.sort_values('Age').reset_index(inplace=True) #Sort values by columns
#%%
df_concat
#%%
df_concat.rename(columns={"name": "Surname", "Age": "Age_ppl"}) #Rename: change col names
df_concat

#%%
#%%
















