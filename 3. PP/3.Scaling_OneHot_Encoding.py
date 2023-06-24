# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 22:59:35 2022

@author: rvamsikrishna
"""
#%%
#File-Contents
#--------------
# One-Hot encoding pd.get_dummies()
# One-Hot encoding using sklearn.preprocessing import OneHotEncoder
# sklearn.preprocessing import LabelEncoder for text to numeric categories
# sklearn.preprocessing import MinMaxScaler
 #sklearn.preprocessing import StandardScaler

#%%
#******************* One-Hot encoding ************************
 #*******************************************************************
#One-Hot encoding the categorical parameters using get_dummies()
one_hot_encoded_data = pd.get_dummies(df, columns = ['marital', 'targeted'])
print(one_hot_encoded_data)

#%%
#One-Hot encoding the categorical parameters using get_dummies()
one_hot_encoded_data = pd.get_dummies(df[['marital', 'targeted']], columns = ['marital', 'targeted'])
print(one_hot_encoded_data)

#%%
#%%
#%%
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
from sklearn.preprocessing import OneHotEncoder
#creating instance of one-hot-encoder
encoder = OneHotEncoder(handle_unknown='ignore')

#perform one-hot encoding on 'marital' column 
encoder_df = pd.DataFrame(encoder.fit_transform(df[['marital']]).toarray())

encoder_df
#%%
#merge one-hot encoded columns back with original DataFrame
final_df = df.join(encoder_df)

#view final df
print(final_df)
    
#drop 'team' column
final_df.drop('marital', axis=1, inplace=True)

#view final df
print(final_df)

#%%
#%%
#%%
#******************* lable encoding ************************
#*******************************************************************

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#%%
# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(type(values))
#%%
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
#%%
# binary encode of integer encoded variable
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
integer_encoded
#%%
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
#%%
# inverse_transform : Convert the data back to the original representation.
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)


#%%
#%%
#%%
#************************ Ordinal encoding ************************
#*******************************************************************

#Using Ordinal Encoder: Required Before Thresholding
#-------------------------------------------------------
#In ordinal encoding, each unique category value is assigned an integer value. 
#For example, “red” is 1, “green” is 2, and “blue” is 3. 
#This is called an ordinal encoding or an integer encoding and is easily reversible.
# Often, integer values starting at zero are used.

# import ordinal encoder from sklearn
from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
  
# Transform the data
col_names = ["Gender","Region_Code","Occupation","Channel_Code","Credit_Product","Is_Active"]
df[col_names] = ord_enc.fit_transform(df[[col_names]])



#%%
#%%
#%%
#******************* Data Normalization ************************
#*******************************************************************
#Data Normalization ( 0-1 Normalization ((x-xmin / xmax-xmin)))
#-------------------------------------------------------------------------

#We apply normalization when the data is skewed on the either axis 
#i.e. when the data does not follow the gaussian distribution.

#In normalization, we convert the data features of different scales
# to a common scale which further makes it easy for the data
# to be processed for modeling. 
#Thus, all the data features tend to have a similar impact on the
# modeling portion.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Changing the working directory to the specified path--
os.chdir("D:/Normalize - Loan_Defaulter")
 
df = pd.read_csv("bank-loan.csv") # dataset

scaler = MinMaxScaler()
 
new_df = pd.DataFrame(scaler.fit_transform(df),
                  columns=data.columns, 
                  index=data.index) 
print(loan)


#%%
#However, suppose we don’t want the income or age to have values like 0. 
#Let us take the range to be (5, 10)

df_scaled = df.copy()
col_names = ['salary', 'Age']
features = df_scaled[col_names]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled[col_names] = scaler.fit_transform(features.values)

#-----------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(5, 10))
df_scaled[col_names] = scaler.fit_transform(features.values)
df_scaled



#%%
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

#As you don’t need to fit it to your test set, you can just apply transformation.
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#%%
#%%
#%%
#%%
#****************Removing features with low variance**************************
#*****************************************************************************

from sklearn.feature_selection import VarianceThreshold

var_thr = VarianceThreshold(threshold = 0.25) #Removing both constant and quasi-constant

var_thr.fit(df.seleselect_dtypes(include=numerics))

var_thr.get_support()

#Picking Up the low Variance Columns:
concol = [column for column in train1.columns 
          if column not in train1.columns[var_thr.get_support()]]

for features in concol:
    print(features)Gender

#Dropping Low Variance Columns:
df.drop(concol,axis=1)

#Don’t forget to convert the columns dtype to integer or flow before 
#applying a threshold.

#Once you identify your low variance columns, you can always reverse the 
#encoding and continue your journey with original data. Also, don't forget to 
#drop the same columns from test data before predicting results! :)
    

#%%

















