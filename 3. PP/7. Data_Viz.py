# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:52:15 2022

@author: rvamsikrishna
"""


#%%
#%%

# For the columns in the dataset which are non-numerical,
# we can use a seaborn count plot to plot a graph against the Churn column.

sns.countplot(x='Churn',data=df, hue='gender',palette="coolwarm_r")

#we can see that gender is not a contributing factor for customer 
#churn in this data set as the numbers of both the genders, that 
#have or haven’t churned, are almost the same.

#%%
sns.countplot(x='Churn',data=df, hue='InternetService')

#We can see that people using Fiber-optic services have a higher churn
# percentage. This shows that the company needs to improve their 
#Fiber-optic service.

#%%
sns.countplot(x='TechSupport',data=df, hue='Churn',palette='viridis')

#Those customers who don’t have tech support have churned more, which is
#pretty self-explanatory. 

#%%
#Tackling numeric data
#-----------------------

ax = sns.histplot(x = 'tenure', hue = 'Churn', data = df, multiple='dodge')

ax.set(xlabel="Tenure in Months", ylabel = "Count")

#%%

sns.histplot(x='MonthlyCharges',hue='Churn',data=df,multiple='dodge')

#%%

