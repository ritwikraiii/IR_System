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
filename4= sys.argv[4]

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

file = open(filename4)
csvreader = csv.reader(file)

#obtain inverted index
files=open(filename2,'rb')
inv_indx=pickle.load(files)

#obtain dictionary of term and length of posting list
docf={}
for term,lists in inv_indx.items():
    docf[term]=len(lists)

#extract top20 doc for each query
querydocls=defaultdict(list)
for row in csvreader:
    if row[0] not in querydocls.keys():
        querydocls[row[0]].append(row[1])
    else:
        if len(querydocls[row[0]])==20:
            continue
        querydocls[row[0]].append(row[1])

file = open(filename3)
csvreader = csv.reader(file)

#skip 1st row
header = []
header = next(csvreader)

#extrat the score from ranked relevant doclist and store in a dict of queryId and docId as key and score as its value
score_dict={}
for row in csvreader:
    score_dict[row[0],row[1]]=row[2]
    
#extract the relevant and non relevant doc for RF for each query and store in seperate dictionary
RF_relevantdoc=defaultdict(list)
RF_nonreldoc=defaultdict(list)
for queryId,docs in querydocls.items():
    for docId in docs:
        if (queryId,docId) in score_dict.keys() and score_dict[queryId,docId]=='2':
            RF_relevantdoc[queryId].append(docId)
        else:
            RF_nonreldoc[queryId].append(docId)
        
#extract the relevant for PsRF for each query and store in seperate dictionary
PsRf_relevantdoc=defaultdict(list)
for queryId,docs in querydocls.items():
    k=0
    for docId in docs:
        k=k+1
        PsRf_relevantdoc[queryId].append(docId)
        if(k==10):
            break

#obtain the queryid and their text in a dictionary
file = open('queries_20.txt', 'r')
query_dict={}
lines = file.readlines()

for line in lines:
    text=re.findall("'(.*?)'", line, re.DOTALL)
    ls=line.split(',')
    queryId=ls[0]
    query_dict[queryId]=text

#total number of queries
nq=len(query_dict.keys())

#total docs
n=len(docIdterms.keys())

#total vocab size
v=len(inv_indx.keys())

#store index from vocab from 0 to v-1 
voc_index={}
ind=0
for word in inv_indx.keys():
    voc_index[word]=ind
    ind=ind+1

#store index for queryId from 0 to nq-1
query_index={}
ind=0
for queryId in query_dict.keys():
    query_index[queryId]=ind
    ind=ind+1    

#tf-idf weighting for Inc.Itc 
#for docId Inc
tf_idfdoc={}
idf=1
for vocab,lists in inv_indx.items():
    for docId in lists:
        tf_idfdoc[docId,vocab]=idf*(1+math.log10(tftd[vocab,docId]))

#for query Itc calulate tf_idf
tf_idfq={}
for queryId,text in query_dict.items():
    for words in text:
        if words in inv_indx.keys():
            df=docf[words]
            idf=math.log10(n/df)
            tf=1
            tf_idfq[queryId,words]=tf*idf

#obtain modified query vector for RF 
def queryVectorRF(alpha,beta,gamma):
    
    q=np.zeros((nq,v))
    for queryId,text in query_dict.items():
        for words in text:
            if words in inv_indx.keys():
                q[query_index[queryId]][voc_index[words]]=alpha*tf_idfq[queryId,words]
                
    for queryId,docs in RF_relevantdoc.items():
        sec_term=np.zeros(v)      #stores sum of vectr for relevant doc
        if len(docs)==0:
            continue
        for docId in docs:
            for word in docIdterms[docId]:
                sec_term[voc_index[word]]=sec_term[voc_index[word]]+((tf_idfdoc[docId,word]*beta)/len(RF_relevantdoc[queryId]))
        q[query_index[queryId]]=np.add(q[query_index[queryId]],sec_term)  
    
    for queryId,docs in RF_nonreldoc.items():
        third_term=np.zeros(v)      #stores sum of vectr for non relevant doc
        if len(docs)==0:
            continue
        for docId in docs:
            for word in docIdterms[docId]:
                third_term[voc_index[word]]=third_term[voc_index[word]]+((tf_idfdoc[docId,word]*gamma)/len(RF_nonreldoc[queryId]))
        q[query_index[queryId]]=np.subtract(q[query_index[queryId]],third_term)  
    return q

#obtain modified query vector for PSRF 
def queryVectorPSRF(alpha,beta):
    q=np.zeros((nq,v))
    for queryId,text in query_dict.items():
        for words in text:
            if words in inv_indx.keys():
                q[query_index[queryId]][voc_index[words]]=alpha*tf_idfq[queryId,words]
                
    for queryId,docs in PsRf_relevantdoc.items():
        sec_term=np.zeros(v)      #stores sum of vectr for relevant doc
        if len(docs)==0:
            continue
        for docId in docs:
            for word in docIdterms[docId]:
                sec_term[voc_index[word]]=sec_term[voc_index[word]]+((tf_idfdoc[docId,word]*beta)/len(PsRf_relevantdoc[queryId]))
        q[query_index[queryId]]=np.add(q[query_index[queryId]],sec_term)  
    return q

def finalresult(q):
    query_doc={}        #key-queryId and retrieved doc list as values
    for i in range(nq):
        score_dict={}
        for docId,terms in docIdterms.items():
            d=np.zeros(v)    #vectorize doc
            for word in terms:
                d[voc_index[word]]=tf_idfdoc[docId,word]
            if norm(d)!=0.0 and norm(q[i])!=0.0:   #check this otherwise null exception will come
                score=np.dot(d,q[i])/(norm(d)*norm(q[i]))
                score_dict[docId]=score
        #sort score_dict by score in descending order
        score_top=dict(sorted(score_dict.items(), key=operator.itemgetter(1),reverse=True))
        #extract top 20 relevant doc
        l=1
        doc_list=[]
        for key,value in score_top.items():
            if value>0:
                doc_list.append(key)
                l=l+1
            if l>20:
                break
        queryId=[k for k, v in query_index.items() if v == i]
        query_doc[queryId[0]]=doc_list      
    return query_doc     

def map20(querydocl,relevantdoc):
    avg_prc20={}
    for queryId,docls in querydocl.items():
    
        #check if queryId is present in relevant doc
        if queryId not in relevantdoc.keys():
            avg_prc20[queryId]=0
            continue
        rel=0
        summ=0
        k=0
        for doc in docls:
            k=k+1
            if doc in relevantdoc.get(queryId):
                rel=rel+1
                summ=summ+(rel/k)
        if rel==0:
            avg_prc20[queryId]=0
        else:
            avg_prc20[queryId]=summ/rel
    map_20=0
    summ=0
    for queryId,prc in avg_prc20.items():
        summ=summ+prc
    map_20=summ/len(avg_prc20.keys())
    return map_20

def ndcg20(querydocl,relevantdoc):
    n_dcg20={}
    for queryId,docls in querydocl.items():
        #check if queryId is present in rankedrelevant doc
        if queryId not in relevantdoc.keys():
            n_dcg20[queryId]=0
            continue
        #store relevance score of first 20 doc scale from 0-2 
        score=[]
        for doc in docls:
            if doc in relevantdoc.get(queryId):
                score.append(int(2))
            else:
                score.append(int(0))
        #calculate actual dcg
        act_dcg=0
        for i in range(0,len(score)):
            if i==0:
                act_dcg=act_dcg+score[i]
            else:
                act_dcg=act_dcg+(score[i]/(math.log2(i+1)))
        if act_dcg==0:
            n_dcg20[queryId]=0
            continue
        #calculate ideal dcg
        score.sort(reverse=True)
        ideal_dcg=0
        for i in range(0,len(score)):
            if i==0:
                ideal_dcg=ideal_dcg+score[i]
            else:
                ideal_dcg=ideal_dcg+(score[i]/(math.log2(i+1)))
        n_dcg20[queryId]=act_dcg/ideal_dcg
        
    ndcg_20=0
    summ=0
    for queryId,ndcg in n_dcg20.items():
        summ=summ+ndcg
    ndcg_20=summ/len(n_dcg20.keys())
    return ndcg_20

#for α = 1 , β = 1 and γ = 0. 5
q=queryVectorRF(1,1,0.5)

qPSRF=queryVectorPSRF(1,1)

#obtain the final result of doc retrieved for all query 
query_docRF=finalresult(q)
query_docPSRF=finalresult(qPSRF)

#calculate map20 and ndcg20 for different α,β and γ
map20RF={}
map20PsRF={}
#for α = 1 , β = 1 and γ = 0. 5
map20RF['1','1','0.5']=map20(query_docRF,RF_relevantdoc)
map20PsRF['1','1','0.5']=map20(query_docPSRF,PsRf_relevantdoc)

ndcg20RF={}
ndcg20PsRF={}
#for α = 1 , β = 1 and γ = 0. 5
ndcg20RF['1','1','0.5']=ndcg20(query_docRF,RF_relevantdoc)
ndcg20PsRF['1','1','0.5']=ndcg20(query_docPSRF,PsRf_relevantdoc)

#for α = 0.5 , β = 0.5 and γ = 0. 5
q=queryVectorRF(0.5,0.5,0.5)
qPSRF=queryVectorPSRF(0.5,0.5)

#obtain the final result of doc retrieved for all query 
query_docRF=finalresult(q)
query_docPSRF=finalresult(qPSRF)

map20RF['0.5','0.5','0.5']=map20(query_docRF,RF_relevantdoc)
map20PsRF['0.5','0.5','0.5']=map20(query_docPSRF,PsRf_relevantdoc)
ndcg20RF['0.5','0.5','0.5']=ndcg20(query_docRF,RF_relevantdoc)
ndcg20PsRF['0.5','0.5','0.5']=ndcg20(query_docPSRF,PsRf_relevantdoc)

#for α = 1, β = 0.5 and γ = 0
q=queryVectorRF(1,0.5,0)
qPSRF=queryVectorPSRF(1,0.5)

query_docRF=finalresult(q)
query_docPSRF=finalresult(qPSRF)

map20RF['1','0.5','0']=map20(query_docRF,RF_relevantdoc)
map20PsRF['1','0.5','0']=map20(query_docPSRF,PsRf_relevantdoc)
ndcg20RF['1','0.5','0']=ndcg20(query_docRF,RF_relevantdoc)
ndcg20PsRF['1','0.5','0']=ndcg20(query_docPSRF,PsRf_relevantdoc)

with open('PB_20_rocchio_RF_metrics.csv', 'w') as csv_file:  
    writer=csv.writer(csv_file,delimiter=',')
    for (alpha,beta,gama) in map20RF.keys():
        writer.writerow([alpha,beta,gama,map20RF[alpha,beta,gama],ndcg20RF[alpha,beta,gama]])

with open('PB_20_rocchio_PsRF_metrics.csv', 'w') as csv_file:  
    writer=csv.writer(csv_file,delimiter=',')
    for (alpha,beta,gama) in map20PsRF.keys():
        writer.writerow([alpha,beta,gama,map20PsRF[alpha,beta,gama],ndcg20PsRF[alpha,beta,gama]])