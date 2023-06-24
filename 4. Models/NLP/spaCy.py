# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:00:38 2019

@author: rvamsikrishna
"""
#https://www.analyticsvidhya.com/blog/2019/06/datahack-radio-ines-montani-matthew-honnibal-brains-behind-spacy/?utm_source=blog&utm_medium=how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python

#https://spacy.io/


#Ines Montani and Matthew Honnibal – The Brains behind spaCy
#**************************************************************

#Linear models used to be quite popular when Matt started working on spaCy because:
#-------------------------------------------------------------------------------------
#These models used a lot of the machine’s memory
#They could be implemented quickly using C or Cython


#The Business Side of spaCy and Prodigy
#*********************************************    

#Ines and Matt settled on annotation tools at the beginning of their journey. 
#And this kept coming up, according to Ines.


#https://prodi.gy/
#Prodigy · An annotation tool for AI, Machine Learning & NLP
#Whether you're working on entity recognition, intent detection or image classification,
# Prodigy can help you train and evaluate your models faster. Stream in your own examples 
#or real-world data from live APIs, update your model in real-time and chain models together
# to build more complex systems.


#So the question was – what were people using and what actually worked for them? Two features stood out:
# Named entity recognition
# Creating labeled data and running experiments


#The Evolution of spaCy (from v1 to v2.1)
#*********************************************    
#The first version of spaCy was built with the linear model technology we saw above. 
#This was back when neural networks were still in their infancy stage, not quite ready 
#to take the machine learning world by storm.

# when neural networks became more and more mainstream, spaCy made the switch from version 1 to 2. 
#Many of the key features in spaCy 2.0 were around the various ML pipeline components 

#spaCy 2.1, the current version, is geared more towards stability and performance.
# One of its stand-out features is dealing with transfer learning and language models


#So what’s next for spaCy?
#-------------------------------
#One of the core uses of spaCy is in information extraction. Basically, going from 
#unstructured text to structured knowledge. What we have in the works is a new component 
#for entity linking – resolving names to knowledge-based entries.



#A few Surprising Use Cases of spaCy
#-------------------------------------
# Extracting information from resumes (PDF parsing)
# Working with network logs


#Future Trends in NLP and spaCy
#---------------------------------
#NLP has come leaps and bounds in the last 12-18 months with the release of 
#breakthrough frameworks like OpenAI’s GPT-2, Google’s BERT, fast.ai’s ULMFiT, among others.

#A change in making NLP models smaller and more efficient. In other words, w
#Which algorithms will be able to scale with ever-larger data sizes and growing datasets?

#We can expect to see a lot of transfer learning aspects in spaCy soon (hello pre-trained models!).
# Spending a lot less time in training our NLP model and not having to wait forever 
#for it to converge sounds perfect to me

#%%
# pip install spacy
# python -m spacy download en_core_web_sm

#import spacy

import en_core_web_sm
nlp = en_core_web_sm.load()

# Load English tokenizer, tagger, parser, NER and word vectors
#nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)


#%%



























