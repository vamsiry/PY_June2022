# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 02:01:52 2019

#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

@author: rvamsikrishna
"""

#Topic Modeling is a technique to extract the hidden topics from large volumes of text.

#This depends heavily on the quality of text preprocessing and the strategy 
#of finding the optimal number of topics. 

#social media, customer reviews of hotels, movies, etc, user feedbacks, news stories, 
#e-mails of customer complaints etc.

#Knowing what people are talking about and understanding their problems and 
#opinions is highly valuable to businesses, administrators, political campaigns

#4. What does LDA do?
#-------------------

#LDA’s approach to topic modeling is it considers each document as
# a collection of topics in a certain proportion. And each topic 
#as a collection of keywords, again, in a certain proportion.

#Once you provide the algorithm with the number of topics, all 
#it does it to rearrange the topics distribution within the documents 
#and keywords distribution within the topics to obtain a good 
#composition of topic-keywords distribution.

#When I say topic, what is it actually and how it is represented?

#A topic is nothing but a collection of dominant keywords that are 
#typical representatives. Just by looking at the keywords, 
#you can identify what the topic is all about.

#The following are key factors to obtaining good segregation topics:
#-------------------------------------------------------------------

#The quality of text processing.
#The variety of topics the text talks about.
#The choice of topic modeling algorithm.
#The number of topics fed to the algorithm.
#The algorithms tuning parameters.


    

#’20 Newsgroups’ dataset and use LDA to extract the naturally discussed topics.
#------------------------------------------------------------------------------

#(LDA) from Gensim package along with the Mallet’s implementation (via Gensim)
#Mallet has an efficient implementation run faster and gives better topics segregation.

#We will also extract the volume and percentage contribution of each topic
# to get an idea of how important a topic is.

#%%
#2. Prerequisites – Download nltk stopwords and spacy model
import nltk; 
#nltk.download('stopwords')

# Run in terminal or command prompt
#python3 -m spacy download en

#%%
#3. Import Packages
#------------------
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline #is a magic function in IPython

#https://stackoverflow.com/questions/43027980/purpose-of-matplotlib-inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='''%(asctime)s : 
    %(levelname)s : %(message)s''', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

#%%
# Prepare Stopwords
#-------------------
# NLTK Stop words

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

#%%
#Import Newsgroups Data
#------------------------

df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
print(df.target_names.unique())
df.head()

#%%
type(df)
#%%
df.size
#%%
df.shape
#%%
df.ndim
#%%
df["content"].ndim 

#%%
# dataframe.size 
size = df.size 
  
# dataframe.shape 
shape = df.shape 
  
# dataframe.ndim 
df_ndim = df.ndim 
  
# series.ndim 
series_ndim = df["content"].ndim 
  
# printing size and shape 
print("Size = {}\nShape ={}\nShape[0] x Shape[1] = {}". 
format(size, shape, shape[0]*shape[1])) 
  
# printing ndim 
print("ndim of dataframe = {}\nndim of series ={}". 
format(df_ndim, series_ndim)) 
# printing size and shape 
print("Size = {}\nShape ={}\nShape[0] x Shape[1] = {}". 
format(size, shape, shape[0]*shape[1])) 
  
#%%

#The output that you get after DF.to_json is a string. 
#So, you can simply slice it according to your requirement and
# remove the commas from it too.

out = df.to_json(orient='records')[1:-1].replace('},{', '} {')

#To write the output to a text file, you could do:
with open('20 Newsgroups.txt', 'w') as f:
    f.write(out)

#%%
with open('temp.json', 'w') as f:
    f.write(df.to_json(orient='records', lines=True))
    
#%%    
df.to_json('temp.json', orient='records', lines=True)

#Direct compression is also possible:
df.to_json('temp.json.gz', orient='records', lines=True, compression='gzip')


#%%
import os
os.getcwd()    

#%%
#Remove emails and newline characters
#----------------------------------------
# Convert to list
data = df.content.values.tolist()

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

pprint(data[:1])


#%%
#Tokenize words and Clean-up text
#------------------------------------
#Gensim’s simple_preprocess() for tokenization nd set deacc=True 
#to remove the punctuations.

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[:1])


#%%
#9. Creating Bigram and Trigram Models
#----------------------------------------
#Gensim’s Phrases model can build and implement the bigrams, 
#trigrams, quadgrams and more. 

#two important arguments to Phrases are min_count and threshold


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])


#%%
#Remove Stopwords, Make Bigrams and Lemmatize
#----------------------------------------
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for 
                          token in doc if token.pos_ in allowed_postags])
    return texts_out


#%%
#Let’s call the functions in order.
#------------------------------------    
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])

#%%
#Create the Dictionary and Corpus needed for Topic Modeling
#--------------------------------------------------------------
#two main inputs to the LDA topic model are the dictionary(id2word)and the corpus

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

#Gensim creates a unique id for each word in the document.(word_id, word_frequency).
#For example, (0, 1) above implies, word id 0 occurs once in the first document

#%%
#what word a given id corresponds to,
id2word[0]

#Or, you can see a human-readable form of the corpus itself.
# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

#%%
#Building the Topic Model
#---------------------------
#alpha and eta are hyperparameters that affect sparsity of the topics. 
#defaults to 1.0/num_topics prior.
#chunksize is the number of documents to be used in each training chunk

#update_every determines how often the model parameters should be updated
# and passes is the total number of training passes.

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


#%%
#View the topics in LDA model
#------------------------------
#LDA model is built with 20 different topics where 
#each topic is a combination of keywords and each keyword contributes
# a certain weightage to the topic.

#see the keywords for each topic and the weightage(importance) 
# of each keyword using lda_model.print_topics()

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())

doc_lda = lda_model[corpus]


#%%
#Compute Model Perplexity and Coherence Score
#------------------------------------------------

#Model perplexity and topic coherence provide a convenient measure 
#to judge how good a given topic model is

# Compute Perplexity  # a measure of how good the model is. lower the better.
print('\nPerplexity: ', lda_model.log_perplexity(corpus)) 

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized,
                                     dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

#%%
# Visualize the topics-keywords
#--------------------------------
#examine the produced topics and the associated keywords.

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis

#Each bubble on the left-hand side plot represents a topic.
# The larger the bubble, the more prevalent is that topic.

#A good topic model will have fairly big, non-overlapping bubbles 
#scattered throughout the chart instead of being clustered in one quadrant.

#A model with too many topics, will typically have many overlaps,
# small sized bubbles clustered in one region of the chart.


#%%
#Building LDA Mallet Model
#-----------------------------

#Gensim provides a wrapper to implement Mallet’s LDA from within 
#Gensim itself. You only need to download the zipfile, unzip it 
#and provide the path to mallet in the unzipped directory to
# gensim.models.wrappers.LdaMallet.

# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
mallet_path = 'path/to/mallet-2.0.8/bin/mallet' # update this path

ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, 
                                             corpus=corpus, 
                                             num_topics=20, 
                                             id2word=id2word)

# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, 
                                           texts=data_lemmatized,
                                           dictionary=id2word,
                                           coherence='c_v')

coherence_ldamallet = coherence_model_ldamallet.get_coherence()

print('\nCoherence Score: ', coherence_ldamallet)


#%%
#give the number of natural topics in the document, finding the
# best model was fairly straightforward.

#How to find the optimal number of topics for LDA?
#------------------------------------------------------

#build many LDA models with different values of number of 
#topics (k) and pick the one that gives the highest coherence value.

#If you see the same keywords being repeated in multiple topics, 
#it’s probably a sign that the ‘k’ is too large.

#The compute_coherence_values() (see below) trains multiple LDA
#models and provides the models and their corresponding coherence scores.

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

#%%
    
# Can take a long time to run.
model_list,coherence_values = compute_coherence_values(dictionary=id2word,
                                                       corpus=corpus,
                                                       texts=data_lemmatized, 
                                                       start=2,
                                                       limit=40,
                                                       step=6)

#%%
# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


#%%
# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
   
    
#%%    
#So for further steps I will choose the model with 20 topics itself.
# Select the model and print the topics
optimal_model = model_list[3]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))

#%%
#%%
#%%
#Finding the dominant topic in each sentence
#-----------------------------------------------

#One of the practical application of topic modeling is to 
#determine what topic a given document is about.

#format_topics_sentences() topic number that has the highest
# percentage contribution in that document.

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)


#%%
#Find the most representative document for each topic
#---------------------------------------------------------
# Group top 5 sentences under each topic

#output actually has 20 rows, one each for a topic. It has the topic number, 
the keywords, and the most representative document.

sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], 
                                                             ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet.head()


#%%
#Topic distribution across documents
#----------------------------------------

#Finally, we want to understand the volume and distribution of
# topics in order to judge how widely it was discussed.

# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics

#%%





























