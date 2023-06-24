# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:24:00 2019

@author: rvamsikrishna
"""

#NLP Pre_processing(tokenization_Normalization_standardization)

#Steps for effective text data cleaning 
#-----------------------------------------

#One of the first steps in working with text data is to pre-process it.

#For example, social media data is highly unstructured – it is an informal communication 
#– typos, bad grammar, usage of slang, presence of unwanted content like URLs, Stopwords, 
#Expressions etc. are the usual suspects.



#NLP : Removing Stopwords and Performing Text Normalization using NLTK and spaCy in Python
#********************************************************************************************


#first step on how to get started with NLP tokenization. tokenization is the most
# basic step to proceed with NLP (text data). This is important because the meaning 
#of the text could easily be interpreted by analyzing the words present in the text.
 
#******************
#Tokenization
#****************
#********************************************************************************************
#********************************************************************************************
#********************************************************************************************

#https://www.analyticsvidhya.com/blog/2019/07/how-get-started-nlp-6-unique-ways-perform-tokenization/?utm_source=blog&utm_medium=how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python

#In lexical analysis, tokenization is the process of breaking a stream of text up into words,
# phrases, symbols, or other meaningful elements called tokens. The list of tokens becomes 
#input for further processing such as parsing or text mining.


#Before processing a natural language, we need to identify the words that constitute 
#a string of characters. That’s why tokenization is the most basic step to proceed 
#with NLP (text data). This is important because the meaning of the text could 
#easily be interpreted by analyzing the words present in the text.


#There are numerous uses of doing this. We can use this tokenized form to:
#Count the number of words in the text
#Count the frequency of the word, that is, the number of times a particular word is present


#Methods to Perform Tokenization in Python
#===========================================

#1. Tokenization using Python’s split() function
#**************************************************
#It returns a list of strings after breaking the given string by the specified separator.
# By default, split() breaks a string at each space


#One major drawback of using Python’s split() method is that we can use only 
#one separator at a time. Another thing to note – in word tokenization, split() 
#did not consider punctuation as a separate token.


#%%
#Word Tokenization
#---------------------
text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""
# Splits at space 
text.split() 

#%%
#Sentence Tokenization
#--------------------
#A sentence usually ends with a full stop (.), so we can use “.” as a separator 
#to break the string:

text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""
# Splits at '.' 
text.split('. ') 


#%%

#2. Tokenization using Regular Expressions (RegEx)
#**************************************************

#Word Tokenization
#---------------------
import re
text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""

tokens = re.findall("[\w']+", text)

tokens

#The “\w” represents “any word character” which usually means alphanumeric (letters, numbers)
# and underscore (_). ‘+’ means any number of times. So [\w’]+ signals that the code 
#should find all the alphanumeric characters until any other character is encountered.

#%%
#Sentence Tokenization
#-------------------------

import re
text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on, Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""
sentences = re.compile('[.!?] ').split(text)
sentences

#Here, we have an edge over the split() method as we can pass multiple separators 
#at the same time. In the above code, we used the re.compile() function wherein 
#we passed [.?!]. This means that sentences will split as soon as any of these 
#characters are encountered.



#%%

#3. Tokenization using NLTK
#******************************

#NLTK contains a module called tokenize() which further classifies into two sub-categories:

#Word tokenize: We use the word_tokenize() method to split a sentence into tokens or words
#Sentence tokenize: We use the sent_tokenize() method to split a document or paragraph into sentences

#Word Tokenization
#-------------------
from nltk.tokenize import word_tokenize 

text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""

word_tokenize(text)

#Notice how NLTK is considering punctuation as a token? Hence for future tasks, 
#we need to remove the punctuations from the initial list.

#%%

#Sentence Tokenization
#-----------------------
from nltk.tokenize import sent_tokenize

text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""

sent_tokenize(text)


#%%

#4. Tokenization using the spaCy library
#*********************************************

#spaCy is an open-source library for advanced Natural Language Processing (NLP). 
#It supports over 49+ languages and provides state-of-the-art computation speed.

#We will use spacy.lang.en which supports the English language.

#Word Tokenization
#-------------------
from spacy.lang.en import English

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""

#  "nlp" Object is used to create documents with linguistic annotations.
my_doc = nlp(text)

# Create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)
token_list


#%%
#Sentence Tokenization
#---------------------

from spacy.lang.en import English

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

# Create the pipeline 'sentencizer' component
sbd = nlp.create_pipe('sentencizer')

# Add the component to the pipeline
nlp.add_pipe(sbd)

text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""

#  "nlp" Object is used to create documents with linguistic annotations.
doc = nlp(text)

# create list of sentence tokens
sents_list = []
for sent in doc.sents:
    sents_list.append(sent.text)
sents_list

#%%

#5. Tokenization using Keras
#********************************

#To perform word tokenization using Keras, we use the text_to_word_sequence method
# from the keras.preprocessing.text class.

from keras.preprocessing.text import text_to_word_sequence
# define
text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""

# tokenize
result = text_to_word_sequence(text)

result

#Keras lowers the case of all the alphabets before tokenizing them. 
#That saves us quite a lot of time as you can imagine!

#%%

#6. Tokenization using Gensim
#**********************************

# It is an open-source library for unsupervised topic modeling and natural language
# processing and is designed to automatically extract semantic topics from a given document.


#Word Tokenization
#--------------------

from gensim.utils import tokenize

text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""

list(tokenize(text))

#%%

#Sentence Tokenization
#-------------------------

#we use the split_sentences method from the gensim.summerization.texttcleaner class:

from gensim.summarization.textcleaner import split_sentences

text = """Founded in 2002, SpaceX’s mission is to enable humans to become a spacefaring civilization and a multi-planet 
species by building a self-sustaining city on Mars. In 2008, SpaceX’s Falcon 1 became the first privately developed 
liquid-fuel launch vehicle to orbit the Earth."""

result = split_sentences(text)

result

#You might have noticed that Gensim is quite strict with punctuation. 
#It splits whenever a punctuation is encountered. In sentence splitting as well, 
#Gensim tokenized the text on encountering “\n” while other libraries ignored it.

#%%
#********************************************************************************************
#********************************************************************************************
#********************************************************************************************


#stopwords removal and text normalization using the popular NLTK, spaCy and Gensim libraries
#==============================================================================================

#https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/?


#What are Stopwords?
#---------------------
#Stopwords are the most common words in any natural language. For the purpose of analyzing
#text data and building NLP models, these stopwords might not add much value to the 
#meaning of the document.

#“There is a pen on the table”. Now, the words “is”, “a”, “on”, and  “the” add no meaning
# to the statement while parsing it. Whereas words like “there”, “book”, and “table” are
# the keywords and tell us what the statement is all about.

#we need to perform tokenization before removing any stopwords.



#Why do we Need to Remove Stopwords?
#------------------------------------
#tasks like text classification, where the text is to be classified into different categories, 
#stopwords are removed or excluded from the given text so that more focus can be given to 
#those words which define the meaning of the text.

#However, in tasks like machine translation and text summarization, removing stopwords is not advisable.

#On removing stopwords, dataset size decreases and the time to train the model also decreases
#improve the performance as there are fewer and only meaningful tokens left.
# Thus, it could increase classification accuracy


#When Should we Remove Stopwords?
#---------------------------------------

#Remove Stopwords
#----------------------
#    Text Classification
#        Spam Filtering
#        Language Classification
#        Genre Classification
#    Caption Generation
#    Auto-Tag Generation


#Avoid Stopword Removal
#------------------------
#    Machine Translation
#    Language Modeling
#    Text Summarization
#    Question-Answering problems


#Different Methods to Remove Stopwords
#==========================================


#1. Stopword Removal using NLTK
#*********************************

#NLTK has a list of stopwords stored in 16 different languages.

import nltk
from nltk.corpus import stopwords
set(stopwords.words('english'))

# sample sentence
text = """He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were."""

# set of stop words
stop_words = set(stopwords.words('english')) 

# tokens of words  
word_tokens = word_tokenize(text) 
    
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 



print("\n\nOriginal Sentence \n\n")
print(" ".join(word_tokens)) 

print("\n\nFiltered Sentence \n\n")
print(" ".join(filtered_sentence)) 

#%%

#2. Stopword Removal using spaCy
#**********************************

#sapCy has a list of its own stopwords that can be imported as STOP_WORDS from 
#the spacy.lang.en.stop_words class.

#An important point to note – stopword removal doesn’t take off the punctuation marks
# or newline characters. We will need to remove them manually.


from spacy.lang.en import English

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

text = """He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were."""

#  "nlp" Object is used to create documents with linguistic annotations.
my_doc = nlp(text)


# Create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)


from spacy.lang.en.stop_words import STOP_WORDS

# Create list of word tokens after removing stopwords
filtered_sentence =[] 

for word in token_list:
    lexeme = nlp.vocab[word]
    if lexeme.is_stop == False:
        filtered_sentence.append(word) 

print(token_list)
print(filtered_sentence)   


#%%
#3. Stopword Removal using Gensim
#**********************************
#While using gensim for removing stopwords, we can directly use it on the raw text. 
#There’s no need to perform tokenization before removing stopwords. This can save us a lot of time.

#We can easily import the remove_stopwords method from the class gensim.parsing.preprocessing. 

from gensim.parsing.preprocessing import remove_stopwords

# pass the sentence in the remove_stopwords function
result = remove_stopwords("""He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, 
and he had indeed the vaguest idea where the wood and river in question were.""")

print('\n\n Filtered Sentence \n\n')
print(result)  


#%%
#2.1 Noise Removal using regular expressions
#**********************************************

# Sample code to remove a regex pattern 
import re 

def _remove_regex(input_text, regex_pattern):
    urls = re.finditer(regex_pattern, input_text) 
    for i in urls: 
        input_text = re.sub(i.group().strip(), '', input_text)
    return input_text

regex_pattern = "#[\w]*"  

_remove_regex("remove this #hashtag from analytics vidhya", regex_pattern)
              
              
#2.1 Noise Removal using split function
#**********************************************
# Sample code to remove noisy words from a text

noise_list = ["is", "a", "this", "..."] 
def _remove_noise(input_text):
    words = input_text.split() 
    noise_free_words = [word for word in words if word not in noise_list] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

_remove_noise("this is a sample text")              


#%%
#%%
#%%

#Introduction to Text Normalization
#=======================================

# text normalization is a process of transforming a word into a single canonical form. 
#This can be done by two processes, stemming and lemmatization. 

#    Lisa ate the food and washed the dishes.
#    They were eating noodles at a cafe.
#    Don’t you want to eat before we leave?
#   We have just eaten our breakfast.
#   It also eats fruit and vegetables.

#the word eat has been used in multiple forms.

#For us, it is easy to understand that eating is the activity here. So it doesn’t
# really matter to us whether it is ‘ate’, ‘eat’, or ‘eaten’ – we know what is going on.

#Unfortunately, that is not the case with machines. They treat these words differently. 
#Therefore, we need to normalize them to their root word, which is “eat” in our example.

#What are Stemming and Lemmatization?
#-----------------------------------------
#Stemming and Lemmatization is simply normalization of words, which means 
#reducing a word to its root form.

#In most natural languages, a root word can have many variants. For example, the word ‘play’ 
#can be used as ‘playing’, ‘played’, ‘plays’, etc. You can think of similar examples (and there are plenty).


#Stemming
#=========
#Stemming is a text normalization technique that cuts off the end or beginning of a word
# by taking into account a list of common prefixes or suffixes that could be found in that word

#It is a rudimentary rule-based process of stripping the suffixes 
#(“ing”, “ly”, “es”, “s” etc) from a word

#Lemmatization
#================
#Lemmatization, on the other hand, is an organized & step-by-step procedure of obtaining
#the root form of the word. It makes use of vocabulary (dictionary importance of words) 
#and morphological analysis (word structure and grammar relations).


#Why do we need to Perform Stemming or Lemmatization?
#=========================================================
#He was driving
#He went for a drive

# both the sentences, driving activity in the past.
#A machine will treat both sentences differently.
# to make the text understandable for the machine,perform stemming or lemmatization.
#Another benefit of text normalization is that it reduces the number of unique words in the text data.
#This helps in bringing down the training time of the machine learning model

#S0, which one should we prefer?
#------------------------------

#Stemming algorithm works by cutting the suffix or prefix from the word. 
#Lemmatization is a more powerful operation as it takes into consideration the 
#morphological analysis of the word.

#Lemmatization returns the lemma, which is the root word of all its inflection forms.

# lemmatization is an intelligent operation that uses dictionaries which are 
#created by in-depth linguistic knowledge. Hence, Lemmatization helps in forming better features.


#Methods to Perform Text Normalization
#==========================================

#1. Text Normalization using NLTK
#************************************

# PorterStemmer() and WordNetLemmatizer() to perform stemming and lemmatization

#Stemming
#-----------
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer

set(stopwords.words('english'))

text = """He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were."""

stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(text) 
    

filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 


Stem_words = []
ps = PorterStemmer()

for w in filtered_sentence:
    rootWord=ps.stem(w)
    Stem_words.append(rootWord)
    
print(filtered_sentence)
print(Stem_words)


#%%

#Lemmatization
#----------------

#Here, v stands for verb, a stands for adjective and n stands for noun. 
#The lemmatizer only lemmatizes those words which match the pos parameter of the lemmatize method.

#Lemmatization is done on the basis of part-of-speech tagging (POS tagging). 

#Now, let’s perform lemmatization on the same text.

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import nltk
from nltk.stem import WordNetLemmatizer

set(stopwords.words('english'))

text = """He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were."""

stop_words = set(stopwords.words('english')) 

word_tokens = word_tokenize(text) 

    
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 

print(filtered_sentence) 


lemma_word = []
wordnet_lemmatizer = WordNetLemmatizer()

for w in filtered_sentence:
    word1 = wordnet_lemmatizer.lemmatize(w, pos = "n")
    word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
    word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a"))
    lemma_word.append(word3)

print(lemma_word)

#%%
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

word = "multiplying" 

lem.lemmatize(word, "v")

stem.stem(word)


#%%

#2. Text Normalization using spaCy
#****************************************

#Unfortunately, spaCy has no module for stemming. To perform lemmatization, check out the below code:
#It provides many industry-level methods to perform lemmatization. 

#make sure to download the english model with "python -m spacy download en"

#Here -PRON- is the notation for pronoun which could easily be removed using regular expressions. 
#The benefit of spaCy is that we do not have to pass any pos parameter to perform lemmatization.


#In Anaconda cmd prompr use to install : python -m spacy download en_core_web_sm 
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp(u"""He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were.""")

lemma_word1 = [] 
for token in doc:
    lemma_word1.append(token.lemma_)
lemma_word1


#%%


#3. Text Normalization using TextBlob
#****************************************

#Just like we saw above in the NLTK section, TextBlob also uses POS tagging to perform lemmatization

#You can read more about how to use TextBlob in NLP here:
#https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/?utm_source=blog&utm_medium=how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python


#It is based on the NLTK library. We can use TextBlob to perform lemmatization.
# However, there’s no module for stemming in TextBlob.

# from textblob lib import Word method 
from textblob import Word 

text = """He determined to drop his litigation with the monastry, and relinguish his claims to the wood-cuting and 
fishery rihgts at once. He was the more ready to do this becuase the rights had become much less valuable, and he had 
indeed the vaguest idea where the wood and river in question were."""

lem = []
for i in text.split():
    word1 = Word(i).lemmatize("n")
    word2 = Word(word1).lemmatize("v")
    word3 = Word(word2).lemmatize("a")
    lem.append(Word(word3).lemmatize())

print(lem)

#%%


# text/Object Standardization
#********************************

#Text data often contains words or phrases which are not present in any standard lexical 
#dictionaries. These pieces are not recognized by search engines and models.


#Some of the examples are – acronyms, hashtags with attached words, and colloquial slangs.
#With the help of regular expressions and manually prepared data dictionaries, this type of 
#noise can be fixed, 
#the code below uses a dictionary lookup method to replace social media slangs from a text.

lookup_dict = {'rt':'Retweet', 'dm':'direct message', 'awsm' : "awesome", 'luv':"love"}

def _lookup_words(input_text):
    words = input_text.split() 
    new_words = [] 
    for word in words:
        if word.lower() in lookup_dict:
            word = lookup_dict[word.lower()]
        new_words.append(word) 
    new_text = " ".join(new_words) 
    return new_text

_lookup_words("RT this is a retweeted tweet by Shivam Bansal")

#%%

#Apart from three steps discussed so far,Noise Removal(stops words),
#Lexicon Normalization(stemming,Lemmatization),text/Object Standardization 
#other types of text preprocessing includes encoding-decoding noise,
# grammar checker, and spelling correction etc.

#https://www.analyticsvidhya.com/blog/2014/11/text-data-cleaning-steps-python/



#%%
#As a typical business problem, assume you are interested in finding:  
#which are the features of an iPhone which are more popular among the fans. 
#You have extracted consumer opinions related to iPhone and here is a tweet you extracted:

#“I luv my &lt;3 iphone &amp; you’re awsm apple. DisplayIsAwesome, sooo happppppy 🙂 
#http://www.apple.com”


#Steps for data cleaning:
#**************************

# 1. Escaping HTML characters:
#***************************
#Data obtained from web usually contains a lot of html entities like &lt; &gt; &amp; 
#which gets embedded in the original data.

#One approach is to directly remove them by the use of specific regular expressions. 

#Another approach is to use appropriate packages and modules (for example htmlparser of Python), 
#which can convert these entities to standard html tags
#For example: &lt; is converted to “<” and &amp; is converted to “&”.

original_tweet = '''I luv my &lt;3 iphone &amp; you’re awsm apple. DisplayIsAwesome, 
                            sooo happppppy 🙂 http://www.apple.com'''

#from html.parser import HTMLParser
import html.parser

html_parser = html.parser.HTMLParser()

tweet = html.unescape(original_tweet)

tweet

#%%
# 2. Decoding data:
#********************

#Text data may be subject to different forms of decoding like “Latin”, “UTF8” etc.
#UTF-8 encoding is widely accepted and is recommended to use.

#open already decodes to Unicode in Python 3 if you open in text mode
#If you want to open it as bytes, so that you can then decode, you need to open with mode 'rb'.

for lines in open('file','rb'):
    decodedLine = lines.decode('ISO-8859-1')
    line = decodedLine.split('\t')
    print(line)    
 

#Luckily open has an encoding argument which makes this easy:
for decodedLine in open('file', 'r', encoding='ISO-8859-1'):
    line = decodedLine.split('\t')

#%%

#I have a browser which sends utf-8 characters to my Python server, but when I retrieve 
#it from the query string, the encoding that Python returns is ASCII. How can I convert 
#the plain string to utf-8?
 
#NOTE: The string passed from the web is already UTF-8 encoded, I just want to
# make Python to treat it as UTF-8 not ASCII.
    
plain_string = "Hi!"
unicode_string = u"Hi!"

type(plain_string), type(unicode_string)    

#%%
# unicode string
string = 'pythön!'
# print string
print('The string is:', string)
# default encoding to utf-8
string_utf = string.encode()
# print result
print('The encoded version is:', string_utf)

#Example 2: Encoding with error parameter
# unicode string
string = 'pythön!'
# print string
print('The string is:', string)
# ignore error
print('The encoded version (with ignore) is:', string.encode("ascii", "ignore"))
# replace error
print('The encoded version (with replace) is:', string.encode("ascii", "replace"))

print('The encoded version (with replace) is:', string.encode("utf-8", "replace"))


#%%

#tweet = original_tweet.encode("ascii", "ignore")
    
tweet = original_tweet.decode("utf-8", "ignore")

tweet

#%%
def make_unicode(input):
    if type(input) != "unicode":
        input =  input.decode('utf-8')
        return input
    else:
        return input

make_unicode(original_tweet)


#%%
html = """\\u003Cdiv id=\\u0022contenedor\\u0022\\u003E \\u003Ch2 class=\\u0022text-left m-b-2\\u0022\\u003EInformaci\\u00f3n del veh\\u00edculo de patente AA345AA\\u003C\\/h2\\u003E\\n\\n\\n\\n \\u003Cdiv class=\\u0022panel panel-default panel-disabled m-b-2\\u0022\\u003E\\n \\u003Cdiv class=\\u0022panel-body\\u0022\\u003E\\n \\u003Ch2 class=\\u0022table_title m-b-2\\u0022\\u003EInformaci\\u00f3n del Registro Automotor\\u003C\\/h2\\u003E\\n \\u003Cdiv class=\\u0022col-md-6\\u0022\\u003E\\n \\u003Clabel class=\\u0022control-label\\u0022\\u003ERegistro Seccional\\u003C\\/label\\u003E\\n \\u003Cp\\u003ESAN MIGUEL N\\u00b0 1\\u003C\\/p\\u003E\\n \\u003Clabel class=\\u0022control-label\\u0022\\u003EDirecci\\u00f3n\\u003C\\/label\\u003E\\n \\u003Cp\\u003EMAESTRO ANGEL D\\u0027ELIA 766\\u003C\\/p\\u003E\\n \\u003Clabel class=\\u0022control-label\\u0022\\u003EPiso\\u003C\\/label\\u003E\\n \\u003Cp\\u003EPB\\u003C\\/p\\u003E\\n \\u003Clabel class=\\u0022control-label\\u0022\\u003EDepartamento\\u003C\\/label\\u003E\\n \\u003Cp\\u003E-\\u003C\\/p\\u003E\\n \\u003Clabel class=\\u0022control-label\\u0022\\u003EC\\u00f3digo postal\\u003C\\/label\\u003E\\n \\u003Cp\\u003E1663\\u003C\\/p\\u003E\\n \\u003C\\/div\\u003E\\n \\u003Cdiv class=\\u0022col-md-6\\u0022\\u003E\\n \\u003Clabel class=\\u0022control-label\\u0022\\u003ELocalidad\\u003C\\/label\\u003E\\n \\u003Cp\\u003ESAN MIGUEL\\u003C\\/p\\u003E\\n \\u003Clabel class=\\u0022control-label\\u0022\\u003EProvincia\\u003C\\/label\\u003E\\n \\u003Cp\\u003EBUENOS AIRES\\u003C\\/p\\u003E\\n \\u003Clabel class=\\u0022control-label\\u0022\\u003ETel\\u00e9fono\\u003C\\/label\\u003E\\n \\u003Cp\\u003E(11)46646647\\u003C\\/p\\u003E\\n \\u003Clabel class=\\u0022control-label\\u0022\\u003EHorario\\u003C\\/label\\u003E\\n \\u003Cp\\u003E08:30 a 12:30\\u003C\\/p\\u003E\\n \\u003C\\/div\\u003E\\n \\u003C\\/div\\u003E\\n\\u003C\\/div\\u003E \\n\\n\\u003Cp class=\\u0022text-center m-t-3 m-b-1 hidden-print\\u0022\\u003E\\n \\u003Ca href=\\u0022javascript:window.print();\\u0022 class=\\u0022btn btn-default\\u0022\\u003EImprim\\u00ed la consulta\\u003C\\/a\\u003E \\u0026nbsp; \\u0026nbsp;\\n \\u003Ca href=\\u0022\\u0022 class=\\u0022btn use-ajax btn-primary\\u0022\\u003EHacer otra consulta\\u003C\\/a\\u003E\\n\\u003C\\/p\\u003E\\n\\u003C\\/div\\u003E"""

print(html.replace("\\/", "/").encode().decode('unicode_escape'))

#%%

original_tweet = '''I luv my &lt;3 iphone &amp; you’re awsm apple. DisplayIsAwesome, 
                            sooo happppppy 🙂 http://www.apple.com'''

tweet = original_tweet.decode("utf-8").encode("ascii","ignore") #throwing error

tweet

#%%

#Apostrophe Lookup: To avoid any word sense disambiguation in text, it is recommended to 
#-------------------------
#maintain proper structure in it and to abide by the rules of context free grammar. 
#When apostrophes are used, chances of disambiguation increases.

#All the apostrophes should be converted into standard lexicons. 
#One can use a lookup table of all possible keys to get rid of disambiguates.

APPOSTOPHES = {"'s" : " is", "'re" : " are", "you're" : " you are"}

original_tweet = """I luv my &lt;3 iphone &amp; you're awsm apple. DisplayIsAwesome, 
                            sooo happppppy 🙂 http://www.apple.com"""
                            
words = original_tweet.split()

reformed = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in words]

reformed = " ".join(reformed)

reformed

#%%


#Removal of Stop-words: When data analysis needs to be data driven at the word level, the commonly
#-------------------------
# occurring words (stop-words) should be removed. One can either create a long list of stop-words 
#or one can use predefined language specific libraries.

#Removal of Punctuations: All the punctuation marks according to the priorities should be dealt with. 
#----------------------------
#For example: “.”, “,”,”?” are important punctuations that should be retained while others need to be removed.

#Removal of Expressions: Textual data (usually speech transcripts) may contain human expressions
#--------------------------
# like [laughing], [Crying], [Audience paused]. These expressions are usually non relevant to
# content of the speech and hence need to be removed. Simple regular expression can be useful in this case.

#Split Attached Words: We humans in the social forums generate text data, which is completely 
#--------------------------
#informal in nature. Most of the tweets are accompanied with multiple attached words like RainyDay,
# PlayingInTheCold etc. These entities can be split into their normal forms using simple rules and regex.

original_tweet = '''I luv my &lt;3 iphone &amp; you’re awsm apple. DisplayIsAwesome, 
                            sooo happppppy 🙂 http://www.apple.com'''

import re

cleaned = " ".join(re.findall("[A-Z][^A-Z]* ", original_tweet))

cleaned
#%%

#Slangs lookup: Again, social media comprises of a majority of slang words. 
#-------------------
#These words should be transformed into standard words to make free text. 
#The words like luv will be converted to love, Helo to Hello. The similar approach of apostrophe 
#look up can be used to convert slangs to standard words. A number of sources are available on the web,
# which provides lists of all possible slangs, this would be your holy grail and you could use 
#them as lookup dictionaries for conversion purposes.

tweet = _slang_loopup(tweet)


# 9 .Standardizing words: Sometimes words are not in proper formats. 
#----------------------------------------------------------------------
#For example: “I looooveee you” should be “I love you”. Simple rules and regular expressions
# can help solve these cases.

tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))

#Removal of URLs: URLs and hyperlinks in text data like comments, reviews, and tweets should be removed
#-----------------------


#Advanced data cleaning:
#-----------------------------
#Grammar checking: Grammar checking is majorly learning based, huge amount of proper text data is learned and models are created for the purpose of grammar correction. There are many online tools that are available for grammar correction purposes.
#Spelling correction: In natural language, misspelled errors are encountered. Companies like Google and Microsoft have achieved a decent accuracy level in automated spell correction. One can use algorithms like the Levenshtein Distances, Dictionary Lookup etc. or other modules and packages to fix these errors.

#%%







































