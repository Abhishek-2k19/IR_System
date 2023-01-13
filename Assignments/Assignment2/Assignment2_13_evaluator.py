#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import sys
import numpy
import glob
import json
import csv
import pickle
import math
import itertools
import operator
import pandas as pd
from itertools import islice
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[2]:


path_to_csv=sys.argv[2]
path_to_gold_standard=sys.argv[1]


# In[38]:


dfRel = pd.read_csv(path_to_gold_standard)
dfRel


# In[39]:


#sorting by iteration to get the highest iteration entry when storing in variable
dfRel.sort_values(by=["iteration"], inplace = True)
dfRel[600:2000]


# In[40]:


qRel ={} #it will be 2d map array of the format query-cord_id

for ind in range(len(dfRel)):
    query=dfRel.loc[ind,'topic-id']
    cord_id=dfRel.loc[ind,'cord-id']
    
    if query not in qRel:
        qRel[query]={}
    qRel[query][cord_id]=dfRel.loc[ind,'judgement']
# print(qRel['2'])
for i in qRel:
    qRel[i]= dict(sorted(qRel[i].items(),key=lambda x:x[1],reverse = True))
qRel


# In[41]:





def compute_values(dfA,q_val):
    qA={}
    for ind in dfA.index:
        if dfA['query_id'][ind] not in qA:
            qA[dfA['query_id'][ind]]=[]
        qA[dfA['query_id'][ind]].append(dfA['document_id'][ind])




    #MAP
    sum10=0
    cnt=0
    for query in qRel:
        numrel=0
        numdoc=0
        sum=0
        for doc in qA[query][0:10]:
            numdoc+=1
            if(doc in qRel[query] and qRel[query][doc]>0):
                numrel+=1
                sum+=(numrel/numdoc)
        if(sum>0):
            sum10 += (sum/numrel)
        cnt+=1
        q_val[cnt]=[]
        if(numrel!=0):
            q_val[cnt].append(sum/numrel)
        else:
            q_val[cnt].append(0)
    map10 = sum10/(len(qRel))

    sum20=0
    cnt=0
    for query in qRel:
        numrel=0
        numdoc=0
        sum=0
        for doc in qA[query][0:20]:
            numdoc+=1
            if(doc in qRel[query] and qRel[query][doc]>0):
                numrel+=1
                sum+=(numrel/numdoc)
        if(sum>0):
            sum20 += (sum/numrel)
        cnt+=1
        if(numrel!=0):
            q_val[cnt].append(sum/numrel)
        else:
            q_val[cnt].append(0)
    map20= sum20/len(qRel)
    
    #ndcg
    
    
    sumndcg10=0
    cnt=0
    for q in qRel:
        arr=[]
        for i in qA[q][0:10]:
            if i in qRel[q]:
                arr.append(qRel[q][i])
            else:
                arr.append(0)
        arr.sort(reverse=True)
        sumIdeal=0
        n=0
        
        for i in arr[0:10]:
            n+=1
            if(n==1):
                sumIdeal+=i
            else:
                sumIdeal+= (i / math.log(n,2))
        sumQ=0
        n=0
        for doc in qA[q][0:10]:
            n+=1
            if doc not in qRel[q]:
                continue
            if(n==1):
                sumQ+= (qRel[q][qA[q][n-1]])
            else:
                sumQ+= (qRel[q][qA[q][n-1]] / math.log(n,2))
        if(sumIdeal==0):
            sumndcg10+=0
            cnt+=1
            q_val[cnt].append(0)
        else:
            sumndcg10+=sumQ/sumIdeal
            cnt+=1
            q_val[cnt].append(0)

    NDCG10= sumndcg10/len(qRel)

    sumndcg20=0
    cnt=0
    for q in qRel:
        arr=[]
        for i in qA[q][0:20]:
            if i in qRel[q]:
                arr.append(qRel[q][i])
            else:
                arr.append(0)
        arr.sort(reverse=True)
        sumIdeal=0
        n=0
        
        for i in arr[0:20]:
            n+=1
            if(n==1):
                sumIdeal+=i
            else:
                sumIdeal+= (i / math.log(n,2))
        sumQ=0
        n=0
        for doc in qA[q][0:20]:
            n+=1
            if doc not in qRel[q]:
                continue
            if(n==1):
                sumQ+= (qRel[q][qA[q][n-1]])
            else:
                sumQ+= (qRel[q][qA[q][n-1]] / math.log(n,2))
        if(sumIdeal==0):
            sumndcg20+=0
            cnt+=1
            q_val[cnt].append(0)
        else:
            sumndcg20+=sumQ/sumIdeal
            cnt+=1
            q_val[cnt].append(sumQ/sumIdeal)

    NDCG20= sumndcg20/len(qRel)
    
    result = {
        "MAP10":map10,
        "MAP20":map20,
        "NDCG10":NDCG10,
        "NDCG20":NDCG20
    }
    return result


# In[43]:


q_valA={}
outputFileA = open("Assignment2_13_metrics_"+path_to_csv[-5]+".txt",'w')
dfA = pd.read_csv(path_to_csv)
resultA = compute_values(dfA,q_valA)
outputFileA.write("Values for each query:"+"\n")
for query in q_valA:
    outputFileA.write(str(query)+":"+" MAP@10 : "+str(q_valA[query][0])+" MAP@20 : "+str(q_valA[query][1])+" NDCG@10 : "+str(q_valA[query][2])+" NDCG@20 : "+str(q_valA[query][3])+"\n")
outputFileA.write("Average values across all queries:"+"\n")
outputFileA.write(str(resultA))
outputFileA.close()


# In[ ]:




