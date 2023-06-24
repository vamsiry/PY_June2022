# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:24:45 2022

@author: rvamsikrishna
"""
#pandas api
#https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html

#File main contents
#-----------------------
# 1. series and df all statistical functions --68 row
# 2. agg() / aggregate() --119 row
# 3. pandas.DataFrame.apply --196 row
# 4. #pandas.DataFrame.transform --295 row
# 5. Pandas apply() vs transform() --331 row
# 6. difference between apply and applymap pandas --458 row
# 7. Groupby and agg in python --500 row


#%%
#File-Contents--Mainly Data Summarizatoin
#-------------------------------------------
#sum(),min(),max(),idxmin(),idxmax(),prod()
#cumsum(),cumprod(),cummin(),cummax(),quantile(),
#count(),value_counts(),df.nunique(),#pd.series.unique(),#pd.series.nunique(),
#mean(),median(),mode(),mad(),sem()(standard error),
#skew(),kurtosis(),
#add(),sub(),mul(),div(),truediv(),floordiv(),mod(),pow(),
#abs(),all(),any(),

#isin(), where(), 
#Series.str.contains
#df.mask() #Replace values where the condition is True.

#Among flexible wrappers (add, sub, mul, div, mod, pow) to 
#arithmetic operators: +, -, *, /, //, %, **.

# Describe()

#list comprehension


# applymap()
# Map()
# filter()
# reduce()
# Lambda()
# first()-- method returns the first n rows, based on the specified value.
# last()-- method returns the last n rows, based on the specified value


#%%
import pandas as pd
import numpy as np

df = pd.read_csv("C://Users//rvamsikrishna//Desktop//PY//Python//PY Data Modeling//marketing_analysis.csv",skiprows=1, low_memory=False)

#url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/HairEyeColor.csv"
#df = pd.read_csv(url)

#%%
df.index
#%%
df.dtypes
#%%
df.sum() #sum of all values column level 
#%%
df.age.sum()
#%%
df[["age","salary"]].sum(axis=1)
#%%
df.sum(numeric_only = True)
#%%
dff = pd.DataFrame({'year': [2012, 2014, 2013, 2014],'month': [1, 4, 7, 4],
                   'sale': [55, 40, 84, 31]})
df2 = dff.set_index(['year', 'month'])

#%%
df2.sum(axis = 0)
#%%
df2.sum(axis = 1)
#%%
df2.sum(axis = 0, level = ['year','month'])
#%%
pd.DataFrame({'Max_freq': [df.age.max()], \
             'Min_freq': [df.age.min()],\
             'Std_freq': [np.std(df.age)]})

#%%
#You can also apply methods to columns of the dataframe:
df2.loc[:,"2005"].mean()
#Note though that in this case you are not applying the mean method to a 
#pandas dataframe, but to a pandas series object:

#%%
type(df2.loc[:,"2005"])
#So, checking the type of the object would give the type of the object:
#%%
df.min(axis = 0,numeric_only = True,skipna = True,)
#note : level : If the axis is a MultiIndex (hierarchical), count 
#along a particular level, collapsing into a Series.

#%%
df.max(numeric_only = True)
#%%
#%%









#%%
#%%
#pandas.DataFrame.agg/aggrigate
#***********************************

numerics = ['number','int16', 'int32', 'int64', 'float16',\
            'float32','float64',np.number,np.int16,np.int32,\
            np.int64,np.float16,np.float32,np.float64,np.number]

df.select_dtypes(include=numerics).agg([len,'count','min','max',\
                'mean', 'median','mad','var','std',\
                'skew','kurtosis'])

#%%
Categorical = ['object','category']

df.select_dtypes(include=Categorical).agg([len,'count','nunique'])

#%%
# agg(): apply more than one aggregation operations to the same dataset over the specified axis.
# agg() is a alias function for aggregate()
df.agg(['count',len,min,max,np.mean])

#%%
df.agg({sum,min,max,len,np.mean})

#%%
df.age.agg([sum,min,max,len,np.mean])

#%%
df.age.agg({sum,min,max,len,np.mean})

#%%
df[['age','previous']].agg(['sum','min'])

#%%
#agg for 3 columns
df[['age','salary','balance']].agg(['count',min,max,len,np.mean]) 

#%%
df.agg({'age':['min','max'],'salary':['mean']})

#%%
df.aggregate({'age':['sum','min'],'previous':'min'})

#%%
df['age'].agg({'sum':sum,'min':min,'max':max,'count':len,'mean':np.mean})
#%%
df.groupby('marital').agg({'salary':['sum', 'max'], 
                         'age':['mean',lambda x: x.max() - x.min()], 
                         'balance':'sum', 
                         'day': lambda x: x.max() - x.min()})

#%%
def max_min(x):
    return x.max() - x.min()

max_min.__name__ = 'Max-Min'

df.groupby('marital').agg({'salary':['sum', 'max'],
          'age': max_min})    

#%%
#%%

    
    
    
    
    
    
    
    

    
    

#%%
#%%
#pandas.DataFrame.apply
#****************************
#%%
#result_type : {‘expand’, ‘reduce’, ‘broadcast’, None},    
df.apply(np.sum, axis=0, result_type = 'broadcast')

#%%
df.apply(np.sum, axis=1)

#%%
df.apply(np.sqrt)

#%%    
df['age'].apply({'sum':sum,'min':min,'max':max,'count':len,'mean':np.mean})

#%%
df[['age','salary','balance']].apply([sum,min,max,np.mean,np.median])

#%%
#creates new col "mean salary" using group by 
mean_purchase =df.groupby('marital')["salary"].mean().rename("User_mean_salary").reset_index() 
print(mean_purchase)
df_1 = df.merge(mean_purchase)
df_1

#%%
##Apply the get_stats() function to each postTestScore bin

df['postTestScore'].groupby(df['categories']).apply(get_stats).unstack()


#%%    
df1 = pd.DataFrame(np.random.rand(4,4), columns=list('abcd'))
df1['group'] = [0, 0, 1, 1]
df1

#%%
#Using apply and returning a Series
def f(x):
    d = {}
    d['a_sum'] = x['a'].sum()
    d['a_max'] = x['a'].max()
    d['b_mean'] = x['b'].mean()
    d['c_d_prodsum'] = (x['c'] * x['d']).sum()
    return pd.Series(d, index=['a_sum', 'a_max', 'b_mean', 'c_d_prodsum'])

df1.groupby('group').apply(f)    
    
#%%
def f_mi(x):
        d = []
        d.append(x['a'].sum())
        d.append(x['a'].max())
        d.append(x['b'].mean())
        d.append((x['c'] * x['d']).sum())
        return pd.Series(d, index=[['a', 'a', 'b', 'c_d'], 
                                   ['sum', 'max', 'mean', 'prodsum']])

df1.groupby('group').apply(f_mi)

#%%
s = pd.Series([20, 21, 12], index=['London', 'New York', 'Helsinki'])    
    
def square(x):
     return x ** 2
#%%
s.apply(square)
#%%
s.apply(lambda x: x ** 2) #Square the values by passing an anonymous function
#%%
def subtract_custom_value(x, custom_value):
     return x - custom_value    
 
s.apply(subtract_custom_value, args=(5,))    
#%%
#Define a custom function that takes keyword arguments and pass these arguments to apply.
def add_custom_values(x, **kwargs):
     for month in kwargs:
         x += kwargs[month]
     return x

s = pd.Series([200, 21, 12], index=['London', 'New York', 'Helsinki'])    
s
#%%
s.apply(add_custom_values, june=30, july=20, august=25)

#%%
#Use a function from the Numpy library.
s.apply(np.log)
#%%







#%%
#%%
#pandas.DataFrame.transform
#****************************
#Call func on self producing a DataFrame with the same 
#axis shape as self.

# DataFrame.transform(self, func, axis=0, *args, **kwargs)[source]¶

df.age.transform(lambda x: x + 1)
#%%
# it is possible to provide several input functions:
df.age.transform([np.sqrt, np.exp])
#%%
#creates new col "mean salary" using  transform
df["User_Mean"] = df.groupby('marital')["salary"].transform('mean') 
df

#%%
df = pd.DataFrame({
    "Date": ["2015-05-08", "2015-05-07", "2015-05-06", "2015-05-05",
        "2015-05-08", "2015-05-07", "2015-05-06", "2015-05-05"],
    "Data": [5, 8, 6, 1, 50, 100, 60, 120]})
    
df    
#%%
df.groupby('Date')['Data'].transform('sum')
#%%
#%%







#%%
#%%
#Pandas apply() vs transform()
#********************************
#https://github.com/BindiChen/machine-learning/blob/main/data-analysis/014-pandas-apply-vs-transform/pandas-apply-vs-transform.ipynb

# 1. Manipulating values
#--------------------------
df = pd.DataFrame({'A': [1,2,3], 'B': [10,20,30] })

def plus_10(x):
    return x+10
#%%
df.apply(plus_10)
#%%
df.transform(plus_10)
#%%
## lambda equivalent
df.apply(lambda x: x+10)
#%%
## lambda equivalent
df.transform(lambda x: x+10)
#%%
#For a single column
df['B_ap'] = df['B'].apply(plus_10)
df
#%%
df['B_tr'] = df['B'].transform(plus_10)
df
#%%
# Difference - 3 main differences
#----------------------------------

#transform() can take a function, a string function, a list of functions,
# and a dict. However, apply() is only allowed a single function.

# transform() cannot produce aggregated results where as apply() can 
#produce aggregated results 


#apply() works with multiple Series at a time. However, transform()
# is only allowed to work with a single Series at a time.
    
#%%
df = pd.DataFrame({'A': [1,2,3], 'B': [10,20,30] })

#1. transform() can takes a function, a string function, a list of functions,
# and a dict. However, apply() is only allowed a function.
#%%
# A string function
df.transform('sqrt')
#%%
# A list of functions
df.transform([np.sqrt, np.exp])
#%%
# A dict of axis labels -> function
df.transform({'A': np.sqrt,'B': np.exp})
#%%
# 2. transform() cannot produce aggregated results
    
# This is working for apply()
df.apply(lambda x:x.sum())
#%%
## but getting error with transform()
df.transform(lambda x:x.sum())
#%%
# 3. apply() works with multiple Series at a time. However, 
#transform() is only allowed to work with a single Series at a time.
def subtract_two(x):
    return x['B'] - x['A']

# Working for apply with axis=1
df.apply(subtract_two, axis=1)    

#%%
# Getting error when trying the same with transform
df.transform(subtract_two, axis=1)
#%%
# apply() works fine with lambda expression
df.apply(lambda x: x['B'] - x['A'], axis=1)
#%%
# Same error when using lambda expression
df.transform(lambda x: x['B'] - x['A'], axis=1)
#%%
# In conjunction with groupby()
#--------------------------------
df = pd.DataFrame({'key': ['a','b','c'] * 3, 'A': np.arange(9),\
                   'B': [1,2,3] * 3})
df  
#%%
#2 differences
#------------------
#transform() returns a Series that has the same length as the input

#apply() works with multiple Series at a time. However, transform() 
#is only allowed to work with a single Series at a time.

#%%
#1. transform() returns a Series that has the same length as the input
def group_sum(x):
    return x.sum()

gr_data_ap = df.groupby('key')['A'].apply(group_sum)
gr_data_ap
#%%
gr_data_tr = df.groupby('key')['A'].transform(group_sum)
gr_data_tr
#%%
#2. apply() works with multiple Series at a time. However, transform() 
#is only allowed to work with a single Series at a time.
def subtract_two(x):
    return x['B'] - x['A']

df.groupby('key').apply(subtract_two)
#%%
## Getting error
df.groupby('key').transform(subtract_two)


#%%
#%%






#%%
#%%
#difference between apply and applymap pandas
#----------------------------------------------
#https://stackoverflow.com/questions/19798153/difference-between-map-applymap-and-apply-methods-in-pandas

#apply() is used to apply a function along an axis of the DataFrame or
# on values of Series. applymap() is used to apply a function to a 
#DataFrame elementwise.

# apply works on a row / column basis of a DataFrame, \
#applymap works element-wise on a DataFrame, and 
#map works element-wise on a Series.

frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'),\
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
#%%
frame
#%%
f = lambda x: x.max() - x.min()
frame.apply(f)

#%%
#Suppose you wanted to compute a formatted string from each floating point 
#value in frame. You can do this with applymap:
format = lambda x: '%.2f' % x
frame.applymap(format)

#%%
frame.applymap(lambda x: x**2)
#%%
frame.applymap(lambda x: len(str(x)))
#%%
#%%








#%%
#%%
#Groupby and agg in python
#*******************************
df.groupby('marital').first()

#%%
df.groupby(['marital','targeted']).first()
#%%
df.groupby(['marital','targeted']).last()
#%%
df.groupby(['marital', 'targeted']).size()
#%%
df.groupby(['marital','targeted']).count().transpose()
#%%
df.groupby(['marital','targeted'])['salary'].mean().unstack()

#%%
df.groupby('response')['salary'].describe().unstack()

#%%
df.groupby('marital')['age'].mean().reset_index() #mean of age col based on marital

#%%
df.groupby('marital').sum().unstack() #sum of all col based on marital

#%%
pd.DataFrame(df.groupby(df.dtypes, axis=1).describe())

#%%
df.groupby('marital')['age'].sum() #sum of age col based on marital
#%%
df.groupby('marital')['age'].mean().add_prefix('mean_')

#%%
df.groupby(['marital','housing'])['age'].sum() #sum of all col based on marital

#%%
df.groupby(['marital','housing'])['age','balance'].sum() #sum of all col based on marital

#%%
df.groupby(['marital','housing'])['age','salary'].agg(['count',\
          min,max,len,np.mean,'std','var'])
    
#%%
df.groupby(['marital','housing'])['age','salary'].agg(['count',\
          min,max,len,np.mean,'std','var']).transpose()    
    
#%%    
df.groupby(['marital','housing'])['age','salary'].agg(['count',\
          min,max,len,np.mean,'std','var']).unstack()
    
#%%
df.groupby(['marital','housing']).agg({"jobedu":"first",\
          "targeted":"count"}).rename(columns={"targeted":"targeted_Count"})

#%%    
df.groupby("marital").agg({"salary":[np.min, np.max, np.mean, np.sum,
          np.count_nonzero, "count", np.std, np.var ]}).round().transpose()
 
#%%
df.groupby(['marital','housing'])['age','salary'].agg({'age':['min','max'],'salary':['mean']})

#%%
df.loc[:, df.columns != 'customerid'].groupby('response').aggregate(max)

#%%
df.loc[:, df.columns != 'customerid'].groupby('response').aggregate(['max','mean','min','nunique'])

#%%
df.loc[df.marital != 'divorced'].groupby('marital')['age','salary','balance'].agg(['count',min,max,len,np.mean])

#%%
df.assign(Gt50 = (df.age < 40)).groupby("marital").agg({"Gt50":"count"})\
.rename(columns ={"Gt50":"<40_age_group_count"})

#%%
# Count for each group
df[df.Eye.isin(['Blue','Hazel']) & (df.Sex=="Male")].groupby(["Eye",\
   "Sex"]).agg({"Freq":"count"}).rename(columns={"Freq":"Count"})

#%%
#lamda function
df.groupby('Sex').Freq.agg({'func1' : 
    lambda x: x.sum(), 'func2' : lambda x: x.prod()})
   
#%%
#A,B,C,D are column names in data frame    
f = {'A':['sum','mean'], 'B':['prod']}
df.groupby('GRP').agg(f).reset_index() 

#%%
f = {'A':['sum','mean'], 'B':['prod'], 'D': lambda g: df.ix[g.index].E.sum()}
df.groupby('GRP').agg(f)

#%%
df.groupby(["marital"]).get_group("divorced")

#%%
#Iterate an operations over groups
# Group the dataframe by regiment, and for each regiment,
for name, group in df.groupby('marital'): 
    # print the name of the regiment
    print(name)
    # print the data of that regiment
    print(group.describe())
    

#%%
#GroupBy dropna
#------------------
#By default NA values are excluded from group keys during the groupby operation. 
#However, in case you want to include NA values in group keys, you could pass 
#dropna=False to achieve it.

df.groupby(by=['marital'],dropna=True)['age'].sum()

#%%
df.groupby("marital").groups

#%%

df1 = pd.DataFrame(np.random.rand(4,4), columns=list('ABCD'))
df1['group'] = [0, 0, 1, 1]
df1

#%%
#A,B,C,D are column names in data frame
f = {'A':['sum','mean'], 'B':['prod']}
df1.groupby('group').agg(f)
#%%
f = {'A':['sum','mean'], 'B':['prod'], 'D': lambda g: df1.ix[g.index].D.sum()}
df1.groupby('group').agg(f)

#%%
cust = lambda g: g[df1.ix[g.index]['C'] < 0.5].sum()
f = {'A':['sum','mean'], 'B':['prod'], 'D': {'my name': cust}}
df1.groupby('group').agg(f)

#%%
#%%
df.columns















#%%
#%%
#Create new columns using groupby and transform
#****************************************************
df['new']=df.groupby('marital')['salary'].transform(lambda x: (x/x.sum())*100)
#%%
df.new.head(5)
#%%
#new column based on AVG_AGE_BY_MARITAL 
df['AVG_AGE_BY_MARITAL'] = df.groupby('marital')['age'].transform(lambda x: x.mean)

#%%
df.groupby('response')['customerid','age','salary','response'].filter(lambda x:x['age'].mean() > 35)

#%%
xx = df.groupby('response')['customerid','age','salary','response'].filter(lambda x:x['age'].mean() > 35)

#%%
xx.groupby('response')['customerid','age','salary','response'].agg({'age':['count','mean'],'salary':['mean']})

#%%
xx.shape

#%%
#filtering baed on mean graterthan 60k
new = df.groupby('response').filter(lambda x:x['salary'].mean() < 60000)

#%%
df['subject_rank'] = df.groupby(['marital'])['age'].rank(ascending=False)


#%%
df.salary.describe()

#%%






















#%%
#%%
#%%
#%%%
#%%
#https://www.datacamp.com/community/tutorials/categorical-data
import seaborn as sns
import matplotlib.pyplot as plt
carrier_count = cat_df_flights['carrier'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of Carriers')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Carrier', fontsize=12)
plt.show()
    



#%%
import pandas as pd

s = pd.Series(["a","b","c","a"], dtype="category")
print(s)

#%%
import pandas as pd

cat = pd.Categorical(['a','b','c','a','b','c','d'], ['c', 'b', 'a'],ordered=True)
print(cat)

#%%
import pandas as pd

s = pd.Series(["a","b","c","a"], dtype="category")
s.cat.categories = ["Group %s" % g for g in s.cat.categories]
print(s.cat.categories)

#%%
#Merging Dictionaries Using **kwargs
#-----------------------------------------
dict1 = {  'Rahul': 4, 'Ram': 9, 'Jayant' : 10 }
dict2 = {  'Jonas': 4, 'Niel': 9, 'Patel' : 10 }
dict3 = {  'John': 8, 'Naveen': 11, 'Ravi' : 15 }
 
print("Before merging")
print("dictionary 1:", dict1)
print("dictionary 2:", dict2)
print("dictionary 3:", dict3)
 
m1 = {**dict1, **dict2, **dict3}
print("after updating :")
print(m1)

#or

#2. Using .update()
m2 = dict1.update(dict2)
print(m2)
#%%
dict1 = {  'Rahul': 4, 'Ram': 9, 'Jayant' : 10 }
dict2 = {  'Jonas': 4, 'Niel': 9, 'Patel' : 10 }
 
print("Before merging")
print("dictionary 1:", dict1)
print("dictionary 2:", dict2)
 
dict3 = dict1.copy()
 
for key, value in dict2.items():
    dict3[key] = value
 
print("after updating :")
print(dict3)

#%%


























