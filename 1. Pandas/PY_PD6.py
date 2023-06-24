# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 15:23:02 2022

@author: rvamsikrishna
"""


#File-Contents
#--------------
# case 1. Creating New col based on existing column
# Case 2. #remove some outliers. beased on np.quantiles
# case 3: Compressing multiple categories to small in multiple columns at a time
# Case 5 : Check the correlation between variables in the data


#%%
import pandas as pd

df = pd.read_csv("C://Users//rvamsikrishna//Desktop//PY//Python//PY Data Modeling//marketing_analysis.csv",skiprows=1, low_memory=False)

#url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/HairEyeColor.csv"
#df = pd.read_csv(url)


#%%
#%%
#%%
#df.insert() #Insert column into DataFrame at specified location.
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

df.insert(0, "col0", pd.Series([5, 6], index=[1, 2]))
#1st values become NAN as original df has index 0,1 by default
df

#%%
#%%
# case 1. Creating New col based on existing column
#---------------------------------------------------
df = pd.DataFrame({'Date':['10/2/2011', '11/2/2011', '12/2/2011', '13/2/2011'], 
                    'Event':['Music', 'Poetry', 'Theatre', 'Comedy'], 
                    'Cost':[10000, 5000, 15000, 2000]}) 
  
#%%
# new column called ‘Discounted_Price’ after applying a 10% 
#discount on the existing ‘Cost’ column.    
# using apply function to create a new column 
df['Discounted_Price'] = df.apply(lambda row: row.Cost - 
                                  (row.Cost * 0.1), axis = 1) 
print(df) 

#%%
#Solution #2: We can achieve the same result by directly 
#performing the required operation on the desired column element-wise.
# create a new column 
df['Discounted_Price2'] = df['Cost'] - (0.1 * df['Cost']) 
df
#%%
#Creating new column using vectorized operation
df['C*D price'] = df.apply(lambda x: x['Cost'] * x['Discounted_Price'], axis=1)
df








#%%
#%%
import pandas as pd
import numpy as np

df = pd.read_csv("C://Users//rvamsikrishna//Desktop//PY//Python//PY Data Modeling//marketing_analysis.csv",skiprows=1, low_memory=False)
df
#%%
df['age2'] = pd.cut(df['age'], bins=[0, 15, 20, 35,100], labels=['Child','teen','Adult','Elder'])
df['age2'].value_counts()
#%%
def Binning(arr):
    bins = np.empty(arr.shape[0])
    for idx, x in enumerate(arr):
        if (x > 0) & (x <= 15):
            bins[idx] = 1
        elif (x > 16) & (x < 20):
            bins[idx] = 2
        elif (x >= 20) & (x < 35):
            bins[idx] = 3
        elif (x >= 35) & (x < 100):
            bins[idx] = 4
        else:
            bins[idx] = 5
    return bins

pd.Series(Binning(df['age'])).head(10)

#%%
def Binning(arr):
    bins = np.empty(arr.shape[0]).tolist()
    for idx, x in enumerate(arr):
        if (x > 0) & (x <= 15):
            bins[idx] = "a"
        elif (x > 16) & (x < 20):
            bins[idx] = "b"
        elif (x >= 20) & (x < 35):
            bins[idx] = "c"
        elif (x >= 35) & (x < 100):
            bins[idx] = "d"
        else:
            bins[idx] = "e"
    return bins

Binning(df['age'])[1:10]

#%%
np.empty(10).tolist()

#%%
bins = [0, 15, 20, 35,100]
df['binned'] = np.searchsorted(bins, df['age'].values)
print (df.binned[1:10])

#%%
#Extract job  & Education in newly from "jobedu" column.
df['job']= df["jobedu"].apply(lambda x: x.split(",")[0])

#%%
df['education']= df["jobedu"].apply(lambda x: x.split(",")[1])

#%%
# You can also add a column containing the average income for each state:
df["Mean_sal_balance"]=df[["salary","balance"]].mean(axis=1) 
df.Mean_sal_balance.head()


#%%
df['day'] = pd.DatetimeIndex(df['dateday']).day #day of the month from 1 to 31.











#%%
#%%
url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/HairEyeColor.csv"
df = pd.read_csv(url)
#%%
df.head()
#%%
# Create new column
df.assign(Eye_Hair =df.Eye +"-"+ df.Hair)

#%%

#%%


titles_dict = {'Capt': 'Other','Major': 'Other', 'Jonkheer': 'Other',
               'Don': 'Other','Sir': 'Other','Dr': 'Other','Rev': 'Other',
               'Countess': 'Other','Dona': 'Other','Mme': 'Mrs','Mlle': 'Miss',
               'Ms': 'Miss','Mr': 'Mr','Mrs': 'Mrs','Miss': 'Miss',
               'Master': 'Master','Lady': 'Other'}

df['Title'] = df['Title'].map(titles_dict)





















#%%
#%%
# Case 2. #remove some outliers. beased on np.quantiles
#--------------------------------------------------------
#remove some outliers. In the salary column, I want to keep 
#the values between the 5th and 95th quantiles.

low = np.quantile(df.Salary, 0.05)
high = np.quantile(df.Salary, 0.95)

df = df[df.Salary.between(low, high)]

    
#%%
#%%
#case 3: Compressing multiple categories to small in multiple columns at a time
#-------------------------------------------------------------------------------
#The dataframe contains many categorical variables. If the number of 
#categories are few compared to the total number values, it is better 
#to use the category data type instead of object. It saves a great 
#amount of memory depending on the data size.

#If the number of categories are less than 5 percent of the total
# number of values, the data type of the column will be changed
# to category.
     
cols = df.select_dtypes(include='object').columns

for col in cols:
    ratio = len(df[col].value_counts()) / len(df)
    if ratio < 0.05:
        df[col] = df[col].astype('category')
    
#%%   
#%%

#%%
#case 4 : Binning the numerical columns 
#-------------------------------        
df['Temp_class'] = pd.qcut(df['Temeratue'], 10, labels=False)  

#or
from sklearn.preprocessing import KBinsDiscretizer  
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
Xt = est.fit_transform(target) 

#or
from sklearn.preprocessing import KBinsDiscretizer  
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
est.fit(target)
Xt = est.transform(target)  

#%%
#%%
# Case 5 : Check the correlation between variables in the data
corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot = True, cmap= 'coolwarm')

#%%