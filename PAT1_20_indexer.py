import glob
import re
import nltk
import os
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import collections
import pickle
import sys
from collections import defaultdict
script = sys.argv[0]
filename = sys.argv[1]

inverted_ind=defaultdict(set)
for files in glob.glob(filename+'/*'):
    for file_name in glob.glob(files+'/*'):
        with open(file_name, 'r',encoding='utf-8') as f:
            
            #extract docId
            head, tail = os.path.split(file_name)
            docId=tail
            
            #extract the text between TEXT tags
            soup = BeautifulSoup(f,'html.parser')
            text=str(soup.find('text').text)   
            
            
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
            
            for t in tokenized_text:
                inverted_ind[t].add(docId)
                   
inv_idx=collections.OrderedDict(sorted(inverted_ind.items()))

fileobj=open('model_queries_20.pth','wb')
pickle.dump(inv_idx,fileobj)
fileobj.close()
