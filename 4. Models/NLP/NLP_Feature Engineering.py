# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:40:55 2019

@author: rvamsikrishna
"""
#https://www.kdnuggets.com/2017/02/natural-language-processing-key-terms-explained.html
#Natural Language Processing (NLP)
#Tokenization
#Normalization
#Stemming
#Lemmatization
#Corpus
#Stop Words
#Parts-of-speech (POS) Tagging
#Statistical Language Modeling
#Bag of Words
#n-grams
#Regular Expressions
#Zipf's Law
#Similarity Measures(Levenshtein ,Jaccard ,Smith Waterman)
#Levenshtein - the number of characters that must be deleted, inserted, or substituted in order to make a pair of strings equal 
#Jaccard - the measure of overlap between 2 sets; in the case of NLP, generally, documents are sets of words 
#Smith Waterman - similar to Levenshtein, but with costs assigned to substitution, insertion, and deletion 

#Syntactic Analysis(Also referred to as parsing, )
#Semantic Analysis
#Sentiment Analysis
#Information Retrieval




#%%
#3.Text to Features (Feature Engineering on text data)
#*********************************************************

def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
    distances = range(len(s1) + 1) 
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances 
    return distances[-1]

print(levenshtein("analyze to the eword vamsi","analyse in of as the in of it"))

#%%

xx = range(len("analyse in of as the in of it") + 1)

type(xx)

for i in xx:
    print(i, end=', ')
    
#%%
    
    

#%%    