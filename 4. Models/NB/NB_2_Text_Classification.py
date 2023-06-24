# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 06:26:29 2022

@author: rvamsikrishna
"""
#https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/

#https://iq.opengenus.org/text-classification-naive-bayes/
#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html


import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.datasets import fetch_20newsgroups

#%%
#step1 - Dataset and Imports and subsetting train and test data
#----------------------------------------------------------
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

# define the training set - sklearn provides us with subset data for training and testing
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
#%%
#Let’s find out how many classes and samples we have:
print("We have {} unique classes".format(len(text_categories)))
print("We have {} training samples".format(len(train_data.data)))

#%%
#Let’s visualize the 5th training sample:
# let’s have a look as some traini  ng data
print(train_data.data[5])    

#%%

#Step2 - Initialize CountVectorizer --Extracting features from text files
#-----------------------------------------------------------------------------
#In order to start using TfidfTransformer you will first have to 
#create a CountVectorizer to count the number of words (term frequency),
# limit your vocabulary size, apply stop words and etc. 
#The code below does just that.
    
#instantiate CountVectorizer()     
count_vect = CountVectorizer(stop_words='english')

# this steps generates word counts for the words in your docs 
X_train_counts = count_vect.fit_transform(train_data.data)

X_train_counts.shape

#%%
#CountVectorizer supports counts of N-grams of words or consecutive
# characters. Once fitted, the vectorizer has built a dictionary 
#of feature indices:
count_vect.vocabulary_.get(u'algorithm')


#%%
##3. Compute the IDF values ---From occurrences to frequencies
#-----------------------------------------------------------------
#Now it’s time to compute the IDFs. Note that in this example, 
#we are using all the defaults with CountVectorizer. You can actually
# specify a custom stop word list, enforce minimum word count, etc

#Now we are going to compute the IDF values by calling 
#tfidf_transformer.fit(word_count_vector) on the word counts we computed earlier.

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(X_train_counts)


#***********************************************************
#To get a glimpse of how the IDF values look, we are going to print 
#it by placing the IDF values in a python DataFrame. The values will be 
#sorted in ascending order.

# print idf values 
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=count_vect.get_feature_names(),columns=["idf_weights"]) 
# sort ascending 
df_idf.sort_values(by=['idf_weights'])

#The lower the IDF value of a word, the less unique it is to any particular document.

#Import Note: In practice, your IDF should be based on a large corpora of text.

#%%
#Step4 - Compute the TFIDF score for your documents
#--------------------------------------------------------
#Once you have the IDF values, you can now compute the tf-idf scores 
#for any document or set of documents. Let’s compute tf-idf scores for 
#the 5 documents in our collection.

# tf-idf scores 
tf_idf_vector = tfidf_transformer.transform(X_train_counts)


#***********************************************************
#Now, let’s print the tf-idf values of the first document to see if it makes sense
feature_names = count_vect.get_feature_names() 
#get tfidf vector for first document 
first_document_vector=tf_idf_vector[0] 
#print the scores 
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"]) 
df.sort_values(by=["tfidf"],ascending=False)


#%%
# step 3 and step4 together in on shot- shortcut
#-------------------------------------------------
#In the above example-code, we firstly use the fit(..) method to fit our
# estimator to the data and secondly the transform(..) method to transform 
#our count-matrix to a tf-idf representation. These two steps can be 
#combined to achieve the same end result faster by skipping redundant 
#processing. This is done through using the fit_transform(..) method 
#as shown below.
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

#%%
#Short cut to shortcut (step 3 and step4 together in on shot- shortcut)
#------------------------------------------------------------------------
#Tfidfvectorizer Usage
#----------------------------

#https://stackoverflow.com/questions/34449127/sklearn-tfidf-transformer-how-to-get-tf-idf-values-of-given-words-in-documen

#With Tfidfvectorizer you compute the word counts, idf and tf-idf values
# all at once. It’s really simple.

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse.csr import csr_matrix #need this if you want to save tfidf_matrix

# settings that you use for count vectorizer will go here 
tf = TfidfVectorizer(input=train_data.data, analyzer='word', ngram_range=(0,1),
                     min_df = 0, stop_words = 'english', sublinear_tf=True)

# just send in all your docs here 
tfidf_matrix =  tf.fit_transform(train_data.data)

#%%
#%%
feature_names = tf.get_feature_names()

doc = 0
feature_index = tfidf_matrix[doc,:].nonzero()[1]
tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])

for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:print(w,s)



#%%
#%%
#Step5 -Model Building
#-------------------------

clf = MultinomialNB().fit(X_train_tfidf, train_data.target)
 
#%%
# Input Data to predict their classes of the given categories
docs_new = ['I have a Harley Davidson and Yamaha.', 'I have a GTX 1050 GPU']

# building up feature vector of our input
X_new_counts = count_vect.transform(docs_new)

# We call transform instead of fit_transform because it's already been fit
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predicting the category of our input text: Will give out number for category
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, train_data.target_names[category]))

#%%
#We now finally evaluate our model by predicting the test data. 
#Also, you'll see how to do all of the tasks of vectorizing, transforming
# and classifier into a single compund classifier using Pipeline.
    
# We can use Pipeline to add vectorizer -> transformer -> classifier all in a one compound classifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
# Fitting our train data to the pipeline
text_clf.fit(train_data.data, train_data.target)

#%%
#Step6 - Evaluation of the performance on the test set¶
#-----------------------------------------------
test_data = fetch_20newsgroups(subset='test',
                               categories = text_categories)

docs_test = test_data.data
# Predicting our test data
predicted = text_clf.predict(docs_test)
print('We got an accuracy of',np.mean(predicted == test_data.target)*100, '% over the test data.')


#%%
#%%
#%%
#Step5 -Model Building Another Qay
#-------------------------
# Build the model

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

sns.set() # use seaborn plotting style

#%%

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



















