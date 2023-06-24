
#Beginners Tutorial for Regular Expressions

#https://www.analyticsvidhya.com/blog/2015/06/regular-expression-python/

#%%
#4 Applications of Regular Expressions

#https://www.analyticsvidhya.com/blog/2020/01/4-applications-of-regular-expressions-that-every-data-scientist-should-know-with-python-code/?utm_source=feed&utm_medium=feed-articles

#%%
#Extracting emails from a Text Document
#Regular Expressions for Web Scraping (Data Collection)
#Working with Date-Time features
#Using Regex for Text Pre-processing (NLP)

#%%
#Extracting emails from a Text Document
#==========================================
#finding/extracting emails and other contact information from large text documents.

import re

# give your filename here
with open("filename.txt", "r") as fp:
  text = fp.read()
  
#or text = "" # replace the “text” with the text of your document and you are good to go
  
re.findall(r"[\w.-]+@[\w.-]+", text)



#%%
##Regular Expressions for Web Scraping (Data Collection)
#=========================================================

#One can simply scrape websites like Wikipedia etc. to collect/generate data.
#But web scraping has its own issues – the downloaded data is usually messy and
# full of noise. This is where Regex can be used effectively!


#Suppose this is the HTML that you want to work on:

html = """<table class="vertical-navbox nowraplinks" style="float:right;clear:right;width:22.0em;margin:0 0 1.0em 1.0em;background:#f9f9f9;border:1px solid #aaa;padding:0.2em;border-spacing:0.4em 0;text-align:center;line-height:1.4em;font-size:88%"><tbody><tr><th style="padding:0.2em 0.4em 0.2em;font-size:145%;line-height:1.2em"><a href="/wiki/Machine_learning" title="Machine learning">Machine learning</a> and<br /><a href="/wiki/Data_mining" title="Data mining">data mining</a></th></tr><tr><td style="padding:0.2em 0 0.4em;padding:0.25em 0.25em 0.75em;"><a href="/wiki/File:Kernel_Machine.svg" class="image"><img alt="Kernel Machine.svg" src="//upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Kernel_Machine.svg/220px-Kernel_Machine.svg.png" decoding="async" width="220" height="100" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Kernel_Machine.svg/330px-Kernel_Machine.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Kernel_Machine.svg/440px-Kernel_Machine.svg.png 2x" data-file-width="512" data-file-height="233" /></a></td></tr><tr><td style="padding:0 0.1em 0.4em">
<div class="NavFrame collapsed" style="border:none;padding:0"><div class="NavHead" style="font-size:105%;background:transparent;text-align:left">Problems</div><div class="NavContent" style="font-size:105%;padding:0.2em 0 0.4em;text-align:center"><div class="hlist">
<ul><li><a href="/wiki/Statistical_classification" title="Statistical classification">Classification</a></li>
<li><a href="/wiki/Cluster_analysis" title="Cluster analysis">Clustering</a></li>
<li><a href="/wiki/Regression_analysis" title="Regression analysis">Regression</a></li>
<li><a href="/wiki/Anomaly_detection" title="Anomaly detection">Anomaly detection</a></li>
<li><a href="/wiki/Automated_machine_learning" title="Automated machine learning">AutoML</a></li>
<li><a href="/wiki/Association_rule_learning" title="Association rule learning">Association rules</a></li>
<li><a href="/wiki/Reinforcement_learning" title="Reinforcement learning">Reinforcement learning</a></li>
<li><a href="/wiki/Structured_prediction" title="Structured prediction">Structured prediction</a></li>
<li><a href="/wiki/Feature_engineering" title="Feature engineering">Feature engineering</a></li>
<li><a href="/wiki/Feature_learning" title="Feature learning">Feature learning</a></li>
<li><a href="/wiki/Online_machine_learning" title="Online machine learning">Online learning</a></li>"""

#It is from a wikipedia page and has links to various other wikipedia pages. 

#The first thing that you can check is what topics/pages does it have link for?
import re
re.findall(r">([\w\s()]*?)</a>", html)


#Similarly, you can extract the links to all these pages by using the following regex:
import re
re.findall(r"\/wiki\/[\w-]*", html)

#Note that if you just combine each of the above link with http://wikipedia.com 
#you’ll be able to navigate to all these wikipedia pages.

#%%
#Working with Date-Time features
#==================================

#since Date and Time have multiple formats available it becomes difficult 
#to work with such data.

date = "2018-03-14 06:08:18"

#extract the “Year” from the date. We can simply use regex to 
#find a pattern where 4 digits occur together:
import re
re.findall(r"\d{4}", date)


#The above code will directly give you the year from the date.
# Similarly, you can extract the month and day information all together in one go!
import re
re.findall(r"(\d{4})-(\d{2})-(\d{2})", date)

#You can do the same thing for extracting the time info like hour, minute and second.
#That was one example of a date format, what if you have a date like the following format?
#12th September, 2019

import re
date2  = "This building was founded on 14th March, 1999. That's is when the official foundation was laid."

re.findall(r"(\d{2})\w+\s(\w+),\s(\d{4})",date2)

#%%
#Using Regex for Text Pre-processing (NLP)
#============================================

#For instance, we can have web scraped data, or data that’s manually collected,
# or data that’s extracted from images using OCR techniques and so on!

#text has a lot of inconsistencies like random phone numbers, web links, 
#some strange unicode characters of the form “\x86…” etc

#We will write a function to clean this text using Regex:
#-------------------------------------------------------
import re
import nltk
nltk.download('stopwords')

# download stopwords list from nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # converting to lowercase
    newString = text.lower()
    # removing links
    newString = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', newString) 
    # fetching alphabetic characters
    newString = re.sub("[^a-zA-Z]", " ", newString)
    # removing stop words
    tokens = [w for w in newString.split() if not w in stop_words]
    # removing short words
    long_words=[]
    for i in tokens:
        if len(i)>=4:                                                 
            long_words.append(i)   
    return (" ".join(long_words)).strip()


#So what did we do here? We basically applied a bunch of operations on our input string:
# removing links
newString = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', newString)    

# fetching alphabetic characters
newString = re.sub("[^a-zA-Z]", " ", newString)

# removing stop words
tokens = [w for w in newString.split() if not w in stop_words]

# removing short words
long_words=[]

for i in tokens:
    if len(i)>=4: 
        long_words.append(i)
return (" ".join(long_words)).strip()

#%%
























































