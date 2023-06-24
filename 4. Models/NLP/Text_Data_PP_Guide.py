# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:48:50 2020

@author: rvamsikrishna
"""

#https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/

#Ultimate guide to deal with Text Data (using Python) –
#for Data Scientists and Engineers
#======================================================

#Table of Contents:

# =============================================================================
#
# 1. Basic feature extraction using text data
# Number of words
# Number of characters
# Average word length
# Number of stopwords
# Number of special characters
# Number of numerics
# Number of uppercase words

# 2. Basic Text Pre-processing of text data
# Lower casing
# Punctuation removal
# Stopwords removal
# Frequent words removal
# Rare words removal
# Spelling correction
# Tokenization
# Stemming
# Lemmatization

# 3. Advance Text Processing
# N-grams
# Term Frequency
# Inverse Document Frequency
# Term Frequency-Inverse Document Frequency (TF-IDF)
# Bag of Words
# Sentiment Analysis
# Word Embedding
# 
# =============================================================================


import pandas as pd
import os

os.chdir("C:\\Users\\rvamsikrishna\\Desktop\\PY\\Python")

#%%
import pandas as pd
train = pd.read_csv('sample-train.csv')

print('\n\nDATA\n\n')
print(train.head())


#%%
#1.1 Number of Words
#==================

# number of words in each tweet.
train['word_count'] = train['tweet'].apply(lambda x: len(str(x).split(" ")))
train[['tweet','word_count']].head()

#%%
#1.2 Number of characters
#============================
train['char_count'] = train['tweet'].str.len() ## this also includes spaces
train[['tweet','char_count']].head()

#Note that the calculation will also include the number of spaces, which you can remove, if required.


#%%
#1.3 Average Word Length
#========================

#average word length of each tweet
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

train['avg_word'] = train['tweet'].apply(lambda x: avg_word(x))
train[['tweet','avg_word']].head()

#%%
#1.4 Number of stopwords
#========================
from nltk.corpus import stopwords
stop = stopwords.words('english')

train['stopwords'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
train[['tweet','stopwords']].head()

#%%
#1.5 Number of special characters
#=================================

#number of hashtags or mentions present in it.

#Here, we make use of the ‘starts with’ function because hashtags 
#(or mentions) always appear at the beginning of a word.

train['hastags'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['tweet','hastags']].head()


#%%
#1.6 Number of numerics
#=========================

train['numerics'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['tweet','numerics']].head()

#%%
#1.7 Number of Uppercase words
#=============================

train['upper'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['tweet','upper']].head()


#%%
#%%
#%%
#2. Basic Pre-processing
#=============================

# Before diving into text and feature extraction, our first step should be 
#cleaning the data in order to obtain better features.

#2.1 Lower case
#================
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['tweet'].head()

#%%
#2.2 Removing Punctuation
#==========================
train['tweet'] = train['tweet'].str.replace('[^\w\s]','')
train['tweet'].head()

#%%
#2.3 Removal of Stop Words
#===========================
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['tweet'].head()

#%%
#2.4 Common word removal
#==========================
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]
freq

#Now, let’s remove these words as their presence will not of 
#any use in classification of our text data.

freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()

#%%
#2.5 Rare words removal
#=======================
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
freq

freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()

#%%
#2.6 Spelling correction
#============================
# help us in reducing multiple copies of words.

#For example, “Analytics” and “analytcs” will be treated as different words
#even if they are used in the same sense. 

from textblob import TextBlob
train['tweet'] = train['tweet'].apply(lambda x: str(TextBlob(x).correct()))

#We should also keep in mind that words are often used in their abbreviated form.
# For instance, ‘your’ is used as ‘ur’. We should treat this before the spelling 
#correction step, otherwise these words might be transformed into any other word

#%%
#2.7 Tokenization
#==================

#Tokenization refers to dividing the text into a sequence of words or sentences. 
#In our example, we have used the textblob library to first transform our tweets
# into a blob and then converted them into a series of words.

TextBlob(train['tweet'][1]).words

#%%
#2.8 Stemming
#============

#Stemming refers to the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple
#rule-based approach. For this purpose, we will use PorterStemmer from the NLTK library.

from nltk.stem import PorterStemmer
st = PorterStemmer()
train['tweet'] = train['tweet'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

#%%
#2.9 Lemmatization
#===================
#Lemmatization is a more effective option than stemming because it converts 
#the word into its root word, rather than just stripping the suffices. 
#It makes use of the vocabulary and does a morphological analysis to obtain the root word

from textblob import Word
train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train['tweet'].head()

#%%
#3. Advance Text Processing
#=============================

#Now, we can finally move on to extracting features using NLP techniques.

#3.1 N-grams
#============

#Unigrams do not usually contain as much information as compared to bigrams and trigrams.

# The longer the n-gram (the higher the n), the more context you have to work with. 
#Optimum length really depends on the application – if your n-grams are too short,
# you may fail to capture important differences. On the other hand, if they are too long, 
#you may fail to capture the “general knowledge” and only stick to particular cases.

#So, let’s quickly extract bigrams from our tweets using the ngrams function of the textblob library.

#TextBlob(train['tweet'][0]).ngrams(2)

train['tweet'] = TextBlob(train['tweet']).ngrams(2)


#%%
#3.2 Term frequency
#===================

#Term frequency is simply the ratio of the count of a word present in a sentence,
# to the length of the sentence.
#TF = (Number of times term T appears in the particular row) / (number of terms in that row)

tf1 = (train['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1

#%%
#3.3 Inverse Document Frequency
#==================================

#intuition behind (IDF) is that a word is not of much use to us if it’s appearing in all the documents.

#Therefore, the IDF of each word is the log of the ratio of the total number of rows to the number of
# rows in which that word is present.

#IDF = log(N/n), where, N is the total number of rows and n is the number of rows in
# which the word was present.

import numpy as np

for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['tweet'].str.contains(word)])))

tf1

#%%
#3.4 Term Frequency – Inverse Document Frequency (TF-IDF)
tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1

#%%
#We can see that the TF-IDF has penalized words like ‘don’t’, ‘can’t’, and ‘use’
# because they are commonly occurring words. However, it has given a high weight 
#to “disappointed” since that will be very useful in determining the sentiment of the tweet.
    
#We don’t have to calculate TF and IDF every time beforehand and then multiply
# it to obtain TF-IDF. Instead, sklearn 

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(train['tweet'])


#%%
#3.5 Bag of Words
#=================
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(train['tweet'])
train_bow


#%%
#3.6 Sentiment Analysis
#=====================
train['tweet'][:5].apply(lambda x: TextBlob(x).sentiment)

train['sentiment'] = train['tweet'].apply(lambda x: TextBlob(x).sentiment[0] )
train[['tweet','sentiment']].head()


#%%
#3.7 Word Embeddings
#=====================

#Word Embedding is the representation of text in the form of vectors. 
#The underlying idea here is that similar words will have a minimum distance between their vectors.

#The first step here is to convert it into the word2vec format.

from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

#Now, we can load the above word2vec file as a model.

from gensim.models import KeyedVectors # load the Stanford GloVe model
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

#Let’s say our tweet contains a text saying ‘go away’. We can easily obtain 
#it’s word vector using the above model:

model['go']
model['away']

#We then take the average to represent the string ‘go away’ in the form of vectors having 100 dimensions.
(model['go'] + model['away'])/2


























