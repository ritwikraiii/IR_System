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
import numpy as np
import math
from numpy import array  
import operator
import csv
from collections import defaultdict
from numpy.linalg import norm 

script = sys.argv[0]
filename1 = sys.argv[1]
filename2= sys.argv[2]
filename3= sys.argv[3]

#obtain dictionary of terms,doc as key and term frequency as value
tftd={}
docIdterms={}     #stores docId along with terms
for files in glob.glob(filename1+'/*'):
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
                if (t,docId) not in tftd.keys():
                    tftd[t,docId]=1
                else:
                    tftd[t,docId]=tftd[t,docId]+1
            for t in tokenized_text:
                if docId not in docIdterms.keys():
                    docIdterms[docId]=[t]
                elif t not in docIdterms[docId]:
                    docIdterms[docId].append(t)

file = open(filename3)
csvreader = csv.reader(file)

#obtain inverted index
files=open(filename2,'rb')
inv_indx=pickle.load(files)

#total vocab size
v=len(inv_indx.keys())

#extract top10 doc for each query
querydocls=defaultdict(list)
for row in csvreader:
    if row[0] not in querydocls.keys():
        querydocls[row[0]].append(row[1])
    else:
        if len(querydocls[row[0]])==10:
            continue
        querydocls[row[0]].append(row[1])

#store index from vocab from 0 to v-1 
voc_index={}
ind=0
for word in inv_indx.keys():
    voc_index[word]=ind
    ind=ind+1

#mapping of index to word
ind_vocab={}
ind=0
for word in inv_indx.keys():
    ind_vocab[ind]=word
    ind=ind+1

#tf-idf weighting for doc
tf_idfdoc={}
idf=1
for vocab,lists in inv_indx.items():
    for docId in lists:
        tf_idfdoc[docId,vocab]=idf*(1+math.log10(tftd[vocab,docId]))

#avg of tf-idf weight for top 10 doc retrieved for each queryId
imp_words=defaultdict(list)
for queryId,docls in querydocls.items():
    q=np.zeros(v)
    for docId in docls:
        d=np.zeros(v)
        for word in docIdterms.get(docId):
            d[voc_index[word]]=tf_idfdoc[docId,word]
        normalized_val=(1/norm(d))
        d=np.multiply(d,normalized_val)
        q=np.add(q,d)
    q=np.multiply(q,0.1)     #take avg of all weights as there are 10 docs so multiply by 1/10=.1
    sorted_vec=np.sort(q)[::-1]
    for i in range(5):
        k=0
        for j in q:
            if j==sorted_vec[i] and (ind_vocab[k] not in imp_words[queryId]):
                imp_words[queryId].append(ind_vocab[k])
                break
            k=k+1
                
with open('PB_20_important_words.csv', 'w') as csv_file:
    writer=csv.writer(csv_file,delimiter=',')
    for queryId in imp_words.keys():
        writer.writerow([queryId,imp_words[queryId][0],imp_words[queryId][1],imp_words[queryId][2],imp_words[queryId][3],imp_words[queryId][4]])