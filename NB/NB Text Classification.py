# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:48:18 2022

@author: rvamsikrishna
"""

#https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a

#https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

#Text Analysis is a major application field for machine learning algorithms. 
#However the raw data, a sequence of symbols (i.e. strings) cannot be fed 
#directly to the algorithms themselves as most of them expect numerical
# feature vectors with a fixed size rather than the raw text documents 
#with variable length.


#In order to address this, scikit-learn provides utilities for the most common 
#ways to extract numerical features from text content, namely:

#1. tokenizing strings and giving an integer id for each possible token, 
#for instance by using white-spaces and punctuation symbols as token separators.

#2. counting the occurrences of tokens in each document.



#In this scheme, features and samples are defined as follows:

#each individual token occurrence frequency is treated as a feature.

#the vector of all the token frequencies for a given document is considered a 
#multivariate sample.

#%%
#Tfidftransformer vs. Tfidfvectorizer

#With Tfidftransformer you will systematically compute word counts 
#using CountVectorizer and then compute the Inverse Document 
#Frequency (IDF) values and only then compute the Tf-idf scores.


#With Tfidfvectorizer on the contrary, you will do all three steps 
#at once. Under the hood, it computes the word counts, IDF values,
# and Tf-idf scores all using the same dataset.

#Note: TfidfVectorizer is used on sentences, while TfidfTransformer
# is used on an existing count matrix, such as one returned by 
#CountVectorizer


#When to use what?
#-----------------------
#If you need the term frequency (term count) vectors for 
#different tasks, use Tfidftransformer.

#If you need to compute tf-idf scores on documents within your
# “training” dataset, use Tfidfvectorizer

#If you need to compute tf-idf scores on documents outside your
# “training” dataset, use either one, both will work.



#%%
#%%
#%%
#Problem Statement
#------------------
#This is a multi-class (20 classes) text classification problem.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

sns.set() # use seaborn plotting style

#%%
# Load the dataset
data = fetch_20newsgroups()# Get the text categories

#%%
print(type(data))

#%%
data.keys()

#%%
#data.values()
print(dir(data))

#%%
#data type of each column
print(type(data.DESCR))
print(type(data.data))
print(type(data.filenames))
print(type(data.target_names))
print(type(data.target))

#%%
data.DESCR
#%%
print(data.data[5])

#%%
data.filenames

#%%
data.target_names

#%%
data.target

#%%
## Get the text categories
text_categories = data.target_names# define the training set

#%%
# define the training set
train_data = fetch_20newsgroups(subset="train", categories=text_categories)# define the test set

#%%
print("\n".join(train_data.data[0].split("\n")[:20]))

#%%
print(train_data.target_names[train_data.target[0]])

#%%
# Let's look at categories of our first ten training data
for t in train_data.target[:10]:
    print(train_data.target_names[t])

#%%
## define the test set
test_data = fetch_20newsgroups(subset="test", categories=text_categories)

#%%
#Let’s find out how many classes and samples we have:
print("We have {} unique classes".format(len(text_categories)))
print("We have {} training samples".format(len(train_data.data)))
print("We have {} test samples".format(len(test_data.data)))

#%%
#Let’s visualize the 5th training sample:
# let’s have a look as some traini  ng data
print(test_data.data[5])

#%%
#As mentioned previously, our data are texts (more specifically, emails) 
#so you should see something like the following printed out:


#The next step consists of building the Naive Bayes classifier and finally 
#training the model. In our example, we will convert the collection of 
#text documents (train and test sets) into a matrix of token counts


#To implement that text transformation we will use the make_pipeline function.
# This will internally transform the text data and then the model will be 
#fitted using the transformed data.


#%%

# Build the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())# Train the model using the training data
model.fit(train_data.data, train_data.target)# Predict the categories of the test data
predicted_categories = model.predict(test_data.data)

#%%
#The last line of code predicts the labels of the test set.
print(np.array(test_data.target_names)[predicted_categories])

#%%
#Finally, let’s build the multi-class confusion matrix to see if the model 
#is good or if the model predicts correctly only specific text categories.

# plot the confusion matrix
mat = confusion_matrix(test_data.target, predicted_categories)

sns.heatmap(mat.T, square = True, annot=True, fmt = "d", 
            xticklabels=train_data.target_names,
            yticklabels=train_data.target_names)

plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()

print("The accuracy is {}".format(accuracy_score(test_data.target, predicted_categories)))

#%%
#5. Bonus: Having fun with the model

#Let’s have some fun using the trained model. Let’s classify whatever sentence we like 😄.

# custom function to have fun
def my_predictions(my_sentence, model):
    all_categories_names = np.array(data.target_names)
    prediction = model.predict([my_sentence])
    return all_categories_names[prediction]

my_sentence = "jesus"
print(my_predictions(my_sentence, model))
#['soc.religion.christian']


my_sentence = "Are you an atheist?"
print(my_predictions(my_sentence, model))
#['alt.atheism']

#%%
#6. Conclusions
    
#We saw that Naive Bayes is a very powerful algorithm for multi-class
# text classification problems.

#%%
#%%
#%%

#%%
#%%
#%%
#%%
#%%
#Bunch Library
#*********************

#Bunch is just like dictionary but it supports attribute type access.
#1) Data Type 
#--Dictionary is in-built type, whereas Bunch is from bunchclass package. bunchclass.

#--Bunch works fine in python 2, but in python 3 it does not work! 
#You import Bunch from sklearn.utils

from sklearn.utils import Bunch

#from bunchclass import Bunch #python2 from sklearn.utils import Bunch

#2) Initialization of bunch does not require '{}', but an explicit function 
#with attributes of the elements you required to be in the bunch.

d1={'a':1,'b':'one', 'c':[1,2,3], '4':'d'}
b1=Bunch(a=1,b='one',c=[1,2,3])

#Also note here the keys of Bunch are attributes of the class. 
#They must be mutable and also follow the conventions for variables.

#3) Accessing the value of the key This is the main difference between the two.
d1['a']
b1['a']
b1.a

#In Bunch, you can access the attributes using dot notations. 
#In dict this is not possible.

#Similarities Both Dictionary and bunch can contain values of any data type. 
#But keys must be mutable. There can be nested dictionaries and nested bunches.

#Utilities of Bunch
#----Bunch() is useful serialization to json.
#----Bunch() is used to load data in sklearn. Here normally a bunch contains
#various attributes of various types(list,numpy array,etc.,).
    
#%% Converting Bunch to dataframe

#1) The Numpy way

#We will first concatenate iris.data and iris.target in numpy
# and convert it into pandas dataframe

import numpy as np
import pandas as pd

num_iris=np.hstack((iris.data,iris.target.reshape(150,1)))
data=pd.DataFrame(num_iris,columns=iris.feature_names+['species'])

data.sample(5)  

#
#%%
#2) pandas way

#“sklearn.utils.bunch to dataframe” Code 

#we will convert iris.data into DataFrame and iris.target into Series 
#and then concatenate them.

feature=pd.DataFrame(iris.data,columns=iris.feature_names)

Target=pd.Series(iris.target,name='Species')

data=pd.concat([feature,Target],axis=1)

data.sample(5) 


#%%
#or
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()

#%%
#or
def bunch_to_dataframe(): 
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer 
    cancer = load_breast_cancer()     
    data = np.c_[cancer.data, cancer.target]
    columns = np.append(cancer.feature_names, ["target"])
    return pd.DataFrame(data, columns=columns)

bunch_to_dataframe()
#%%
#%%
#%%


