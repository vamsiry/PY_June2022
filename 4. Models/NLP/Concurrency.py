# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:39:05 2019

@author: rvamsikrishna
"""



import pandas as pd
df = pd.read_excel("C:\\Users\\rvamsikrishna\\Desktop\\Backup\\Projects\\Concurrency\\concc\\concurr.xlsx")

#%%

df = df.sort_values(by=["Ldap","Start_time"]).reset_index(drop=True)

result = []
for i in df["Ldap"].drop_duplicates():
    singlename = df[df["Ldap"]==i].reset_index(drop=True)
    print(i)
    for j in range(len(singlename)):
        result.append((len(singlename[(singlename["Start_time"]<= singlename.iloc[j]["Close_time"])
        &(singlename["Close_time"] >= singlename.iloc[j]["Start_time"])])))

#%%    
df["Concurrency"] = pd.DataFrame(result)    

#%%

df["Concurrency"].describe() 

#%%

df['Concurrency'].apply(str).describe() 

#df["Concurrency"].describe() 

#%%
import os
os.getcwd()

#type(df)

#df.to_csv(r'Path where you want to store the exported CSV file\File Name.csv')

df.to_csv(r'C:\\Users\\rvamsikrishna\\Desktop\\Backup\\Projects\\Concurrency\\concc\\Py_Con.csv',index=False)

#%%

