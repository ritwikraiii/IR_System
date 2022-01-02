import glob
import re
import nltk
import os
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import collections
import pickle
import sys

script = sys.argv[0]
filename = sys.argv[1]

file1 = open(filename, 'r')
query_dict={}
lines = file1.readlines()
for line in lines:
    queryId=re.findall("<num>(.*?)</num>", line, re.DOTALL)
    text=re.findall("<title>(.*?)</title>", line, re.DOTALL)
    
    
    if(len(queryId)!=0):
        lqueryId=int(queryId[0])
    if(len(text)!=0):
        text=str(text)
        
        #convert to lower case
        text=text.lower()
    
        #remove punctuation marks such as , or .
        text = re.sub(r'[^\w\s]','',text)
        
        #convert the words into tokens
        tokenized_text=word_tokenize(text)
        #remove stopwords
        tokenized_text=[t for t in tokenized_text if t not in stop_words]
            
        #perform lemmatization 
        tokenized_text=[WordNetLemmatizer().lemmatize(t) for t in tokenized_text]
        query_text=[t for t in tokenized_text]
        query_dict[lqueryId]=query_text
        
with open("queries_20.txt", 'w') as f: 
    for ids,text in query_dict.items(): 
        f.write('%d,%s\n' %(ids, text))
