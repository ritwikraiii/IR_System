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
filename1 = sys.argv[1]
filename2 = sys.argv[2]
files = open(filename2, 'r')

query_dict={}
lines = files.readlines()

for line in lines:
    text=re.findall("'(.*?)'", line, re.DOTALL)
    ls=line.split(',')
    queryId=int(ls[0])
    query_dict[queryId]=text

files=open(filename1,'rb')
inv_indx_obt=pickle.load(files)

#index the docId from 0 to n-1
docId_indx={}
ind=0
for term,ids in inv_indx_obt.items():
    for docId in ids:
        if docId not in docId_indx.keys():
            docId_indx[docId]=ind
            ind=ind+1
            
#modify posting list of inverted index to the alloted index and sort them
inv_indx={}
for term,ids in inv_indx_obt.items():
    ls=[]
    for docId in ids:
        ls.append(docId_indx[docId])
    ls.sort()
    inv_indx[term]=ls

def intersect(list1,list2):
    i=0
    j=0
    m=len(list1)
    n=len(list2)
    al=[]
    while i<m and j<n:
        if list1[i]<list2[j]:
            i=i+1
        elif list1[i]>list2[j]:
            j=j+1
        else:
            al.append(list1[i])
            i=i+1
            j=j+1
    return al

#result           
result_dict={}
for ids,text in query_dict.items():
    
    #for storing terms and doc frequency
    trm_len={}
    for words in text:
        if words in inv_indx.keys():
            trm_len[words]=len(inv_indx.get(words))
    
    #sort according to length i.e values of trm_len dictionary
    trm_sorted={k: v for k, v in sorted(trm_len.items(), key=lambda item: item[1])}
    
    #first merge two lists with min length
    list1=inv_indx.get(list(trm_sorted.keys())[0])
    list2=inv_indx.get(list(trm_sorted.keys())[1])
    ls=intersect(list1,list2)
    
    #remove first two text of queries from dictionary
    del trm_sorted[list(trm_sorted.keys())[0]]
    del trm_sorted[list(trm_sorted.keys())[0]]
    
    for trm,length in trm_sorted.items():
        ls=intersect(ls,inv_indx.get(trm))
    
    #store final list of doc in separate dict along with query ids as key
    result_dict[ids]=ls

#storing results in a separate file
with open("PAT1_20_results.txt", 'w') as f: 
    for ids,lists in result_dict.items():
        lists=[str(k) for t in lists for k,v in docId_indx.items() if v==t]
        doclist = '  '.join(lists)
        f.write('%d:%s\n' %(ids,doclist))
