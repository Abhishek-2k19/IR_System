#!/usr/bin/env python
# coding: utf-8

# Importing all requirements and reading all the paths and pre-processing

# In[1]:


import nltk
import numpy
import glob
import json
import csv
import pickle
import math
import itertools
import operator
import gc
import os
import sys


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


# reading all the supplied paths
path_to_dataset = sys.argv[1]
path_to_inverted_idx = sys.argv[2]
path_to_gold_standard_ranked_list = sys.argv[3]
path_to_listA = sys.argv[4]
path_to_queries = "./Data/queries.csv"


# Functions and codes for generating TF-IDF for corpus and query and ranking them based on similarity

# In[3]:


# mapping the data
reader = csv.DictReader(open('./Data/id_mapping.csv'))
id_mapping={}

for row in reader:
    id_mapping[row['paper_id']] = row['cord_id']


# In[4]:


# reading inverted index from part 1
inverted_index ={}

with open (path_to_inverted_idx,'rb') as pick:
    inverted_index.update(pickle.load(pick)) # retrieving it from stored pickle file


# In[5]:


# generating df
counter={}
for key in inverted_index:
    counter[key]= len(inverted_index[key])


# In[6]:


# Reducing vocabulary to 20,000 (by taking top 20,000 terms of original df sorted by their df values)
sorted_df = dict( sorted(counter.items(), key=operator.itemgetter(1),reverse=True))
df = dict(itertools.islice(sorted_df.items(), 20000))
# print(df["result"])


# In[7]:


# preprocessing function for returning tokens from the given text
def preprocess(sentence):
    sentence = sentence.lower()          
    sentence = re.sub(r'[^\w\s]', '', sentence)
    
    text_tokens = word_tokenize(sentence)
    tokens_without_sw = [lemmatizer.lemmatize(word) for word in text_tokens if not word in stopwords.words()]
    
    return tokens_without_sw


# In[8]:


# reading dataset again to make the tf frequency
files = glob.glob(path_to_dataset+'/*', recursive = True)
N= len(files)


# In[9]:


# generating the tf again from the corpus

# breaking the files in chunk of 5000 for ease of processing 
chunckSize = 5000
file_data = [files[i * chunckSize:(i + 1) * chunckSize] for i in range((len(files) + chunckSize - 1) // chunckSize )]
del files
gc.collect()

# making a temporary folder for storing tf chunks, which we will merge at the end
dir_path = './temp_tf'
if(not os.path.exists(dir_path)):
        os.mkdir(dir_path)


# iterating on each document of each chunk, making tf for each chunk and saving it to disk and clearing memory and then going to next chunk  
for itr,fileChunk in enumerate(file_data):
    tf = {}
    for file in fileChunk:
        # open file
        f= open(file)
        data = json.load(f)
        sentence=""
        # preprocess and generate tokens
        for obj in data['abstract']:
            sentence += obj['text'] +" "
        sentence = preprocess(sentence)
        doc_id = id_mapping[data['paper_id']]
        # initialize the doc-id in tf file with empty dictionary
        tf[doc_id]={}
        #iterate on each word in tokens obtained
        for word in sentence:
            # Add only if we have taken that word in our limited vocabulary 
            if word in df:
                # if word is not yet initialized in tf[doc_id]. then initialize it with zero
                if word not in tf[doc_id]:
                    tf[doc_id][word] = 0
                # increment the tf by 1
                tf[doc_id][word]+=1
        # close the file
        f.close()
        # delete all the temporary variables
        del data
        del sentence
        # free up memory by calling garbage collector
        gc.collect()
    # save the file in format tf<itr>.bin in the temp_tf folder
    fileName = "tf"+str(itr)+".bin"
    pick_path = os.path.join(dir_path,fileName)
    with open (pick_path, 'wb') as pick:
        pickle.dump(tf, pick)
    # delete tf variable and free up the memory 
    del tf
    gc.collect()


# In[ ]:


# merging the saved tf chunks

# reading paths of chunk files
new_temp_tfs = glob.glob('./temp_tf/*', recursive = True)

print("merging the tf chunks...")

tf = {}
for itr,file_path in enumerate(new_temp_tfs):
    # dictionary to store chunk data
    dict_temp = {}
    with open (file_path,'rb') as pick:
        dict_temp.update(pickle.load(pick))
    # updating tf with dict_temp
    tf.update(dict_temp)
    # deleting dict_temp and freeing up memory
    del dict_temp
    gc.collect()

# saving the file for future reference
print("saving the file...")
with open ("./TF.bin", 'wb') as pick:
    pickle.dump(tf, pick)


## uncomment this code & comment everything in this and the cell above if you already have binary file for tf saved in disk
# tf = {}
# with open ("./finalTF_file.bin",'rb') as pick:
#     tf.update(pickle.load(pick))


# In[10]:


# similarly generating the vector space for query, following lnc.ltc

# reading query file and mapping the queries
reader = csv.DictReader(open('./Data/queries.csv'))
query={}
for row in reader:
    query[row['topic-id']] = row['query']

# generating tf for queries in the same way as we did for corpus
tf_query={}
for index in query:
    sentence = preprocess(query[index])
    tf_query[index]={}
    for word in sentence:
        if word in df:
            if word not in tf_query[index]:
                tf_query[index][word] = 0
            tf_query[index][word]+=1


# In[11]:


# function to normalize tf_idf
def normalize(df):
    for doc_id in df:
        sum=0;
        for w in df[doc_id]:
            sum+= df[doc_id][w]**2
        if(sum==0):
            continue
        for w in df[doc_id]:
            df[doc_id][w] /= math.sqrt(sum)


# In[12]:


# generating tf_idf for corpus
tf_idf={}
for doc_id in tf:
    tf_idf[doc_id]={}
    for word in tf[doc_id]:
        if(tf[doc_id][word] ==0):
            tf_idf[doc_id][word]=0
        else:
            tf_idf[doc_id][word]= 1 + math.log(tf[doc_id][word],10)

# normalize the tf_idf
normalize(tf_idf)


# In[13]:


# generating query tf_idf following lnc.ltc scheme

query_tf_idf={}
for index in tf_query:
    query_tf_idf[index]={}
    for word in tf_query[index]:
        if(tf_query[index][word] ==0):
            query_tf_idf[index][word]=0
        else:
            query_tf_idf[index][word]= (1 + math.log(tf_query[index][word],10))* math.log(N/df[word])


# In[14]:


# function to generate cosine similarity code and ranking the query (returning top 50)
def rankFunc(query_tf_idf):
    # normalizing the query_tf_idf
    normalize(query_tf_idf)
    score={}

    # calculating cosine similarity
    for q in query_tf_idf:
        score[q]={}
        for doc_id in tf_idf:
            score[q][doc_id]=0
            for word in tf_idf[doc_id]:
                if word in query_tf_idf[q]:
                    score[q][doc_id]+= (tf_idf[doc_id][word] * query_tf_idf[q][word])
            
    # taking top 50 results
    for q in query_tf_idf:
        score[q]=dict( sorted(score[q].items(), key=operator.itemgetter(1),reverse=True))
        score[q]=dict(itertools.islice(score[q].items(), 50))
    return score


# In[15]:


# calculating initial rankings to get the scores and tf_idf for original queries
intial_ranking = rankFunc(query_tf_idf) # initial_ranking[query-id] gives top 50 retrieved docs with their scores as key value pairs
# query_tf_idf.keys()


# Taking top 20 documents from the list of documents retrieved for all the queries in part 2 

# In[17]:


# reading the retrieved rank list
rankedListFile = open(path_to_listA,'r')
rankedList = csv.reader(rankedListFile)

# taking top 20 documents from the retrieved ranked list
top20_retrieved = {} # top20_retrieved[query-id] = [list of top 20 doc id], everything in string, including query-id 
for itr,result in enumerate(rankedList,0):
    # taking top 20 results from ranked list retrieved in part 2
    if(itr%50 >=1 and itr%50 <=20):
        if(itr%50==1):
            # initialize the top20_retrieved with the query-id on its first occurence to store the list of top 20 results
            top20_retrieved[result[0]] = []
        top20_retrieved[result[0]].append(result[1])

# closing the file
rankedListFile.close()


# Reading gold standard rank list, removing duplicates and converting it to form - goldStandard_dict[query-id][doc-id] = relevance, for easy access

# In[18]:


# reading the gold standard ranked list
goldStandard_rankedListFile = open(path_to_gold_standard_ranked_list,'r')
goldStandard_rankedList = csv.reader(goldStandard_rankedListFile)
goldStandard = []
for i,row in enumerate(goldStandard_rankedList):
    if i==0:
        continue
    goldStandard.append(row)
goldStandard_rankedListFile.close()

# function to sort according to the value at first index of passed parameter
def sortFunc(e):
    return int(e[0])


# sorting according to the query (since we don't know if qrels is sorted or not)
goldStandard.sort(key=sortFunc)

# creating a visited array and initializing it with 0, just to keep avoid revisiting enteries by keeping track of enteries that we already visited
vis = [0]*(len(goldStandard)+1)


# removing duplicates
for idx,row in enumerate(goldStandard,1):
    
    dup_vals=[] # contains tuples (iteration,index) for multiple occurence of same document for given query
    if(vis[idx]==1):
        continue # don't visit again if it is already visited
    vis[idx]=1 # mark this as visited
    contains_dup=False # flag which turns true if multiple occurences are present
    
    #iterating on all the documents from index idx
    for curIdx,items in enumerate(goldStandard[idx:],idx+1):
        
        # break if we get to next query
        if items[0]!=row[0]:
            break
            
        # checking if multiple occurences exist
        if items[2]==row[2]:
            vis[curIdx] = 1 # marking it visited
            contains_dup = True # turning flag to true
            tup = (items[1],curIdx) # creating tuple of (iteration(string),index)
            dup_vals.append(tup) # appending tupple to the dup_vals list
    if(contains_dup):
        tup = (row[1],idx) #creating tupple for current row 
        dup_vals.append(tup) 
        dup_vals.sort(reverse=True) #sorting based on iteration value (descending order)
        final_elem = goldStandard[dup_vals[0][1]-1] # final element which needs to be taken is the one with highest ieration value
        
        # making all the multiple existence of same relevance and itertion value (duplicating the most relevant entry)
        for itr1,elem in enumerate(dup_vals):
            for itr2,value in enumerate(final_elem):
                goldStandard[dup_vals[itr1][1]-1][itr2] = value

# removing duplicate entries
final_goldStandard = [row for i,row in enumerate(goldStandard) if row not in goldStandard[:i]]


# In[19]:


# creating dictionary of final_goldStandard for easy retrieval
goldStandard_dict = {} # goldStandard_dict[query-id][doc-id] = relevance, (everything except relevance is in string, including query id)
for row in final_goldStandard:
    if row[0] not in goldStandard_dict:
        goldStandard_dict[row[0]] = {}
    goldStandard_dict[row[0]][row[2]] = int(row[3])


# Functions for Rocchio and merging and multiplying two dictionaries

# In[20]:


# code for merging two dictionaries
def Merge(dict1, dict2):
    temp=dict2.copy()
    for key in dict1:
        if(key not in temp):
            temp[key]=0
        temp[key]+=dict1[key]
      
    return temp

# code for multiplying a dictionary with some constant 
def prod(Dict,val):
    temp = Dict.copy()
    for key in temp:
        temp[key]=val*temp[key];
    return temp

# rocchio's algorithm for finding the shifted queries
def rocchio(alpha, beta, gamma, query, relevant, nonrelevant):
    modified_query={}
    modified_query=prod(query,alpha)
    for doc_id in relevant:
        modified_query=Merge(modified_query,prod(relevant[doc_id],beta*1/len(relevant)))
    for doc_id in nonrelevant:
        modified_query=Merge(modified_query,prod(nonrelevant[doc_id],(-1*gamma)/len(nonrelevant)))    
    return modified_query


# Defining sets of alpha, beta and gamma, which needs to be used for Rocchio

# In[21]:


# defining alpha, beta and gamma values for rocchio
alpha = [1,0.5,1]
beta = [1,0.5,0.5]
gamma = [0.5,0.5,0]


# Defining functions for calculating NDCG and MAP at given number of documents

# In[22]:


# defining NDCG function
def NDCG(query_id,new_ranking,numberOfDocs):
    # stores the relevance of queries in the order in which it is retrieved
    lst_actual = []
    for itr,doc_id in enumerate(new_ranking,1):
        # if doc_id,query is in goldstandard, then add its relevance to list, if not then by default append 0
        if(doc_id in goldStandard_dict[query_id].keys()):
            lst_actual.append(goldStandard_dict[query_id][doc_id])
        else:
            lst_actual.append(0)
        # break when we reach the number of documents upto which we need NDCG
        if(itr==numberOfDocs):
            break
    # generating a list (ground truth relevance) which contains the sorted (decreasing order) form of actual list
    lst_sorted = sorted(lst_actual,reverse=True)
    sum_actual = 0 # contains actual DCG
    sum_gt = 0 # contains ground truth DCG
    for i in range(1,numberOfDocs+1):
        # for first element, add directly relevance, for rest of the elements, add relevance/log(position,2)
        if i==1:
            sum_actual += lst_actual[i-1]
            sum_gt += lst_sorted[i-1]
            continue
        sum_actual += (lst_actual[i-1]/ math.log(i,2))
        sum_gt += (lst_sorted[i-1]/math.log(i,2))
        
    # no relevant query in corpus, so NDCG couldn't be calculated    
    if sum_gt==0:
        print("NDCG for query "+query_id+" couldn't be calculated due to no relevant documents in ground truth metric!")
        return -1
    return sum_actual/sum_gt


# In[23]:


# defining MAP function
def meanAP(query_id,new_ranking,numberOfDocs):
    sum_ap = 0 # sum of average precision
    relv_cnt=0 # sum of relevant count
    for itr,doc_id in enumerate(new_ranking,1):
        # At each relevant doc, increase count of relevance by 1 and add the relevance to sum of average precision
        if(doc_id in goldStandard_dict[query_id] and goldStandard_dict[query_id][doc_id]>0):
            relv_cnt+=1
            sum_ap+= (relv_cnt/itr) 
        # break when we reach the number of documents upto which we need NDCG
        if(itr==numberOfDocs):
            break
    
    # if we got some relevant docs, then return MAP, else return 0
    if(relv_cnt!=0):
        return sum_ap/relv_cnt
    else:
        return 0


# Relevance feedback

# In[24]:


# vector for storing final results
final_results = [["alpha","beta","gamma","mAP@20","NDCG@20"]]

# modifying query vector using rocchio for three sets of alpha, beta & gamma
for i in range(3):
    curAlpha = alpha[i]
    curBeta = beta[i]
    curGamma = gamma[i] 
    modified_query_tf_idf={}
  # generating modified query using rocchio
    for q_results in range(1,len(top20_retrieved)+1):
        relevant_docs={} # stores tf_idf of relevant docs in retrieved results for each query
        nrelevant_docs={} # stores tf_idf of non-relevant docs in retrieved results for each query
        modified_query_tf_idf[str(q_results)]={}
        for doc in top20_retrieved[str(q_results)]:
            relevant_docs[doc]={}
            nrelevant_docs[doc]={}
            if(doc in goldStandard_dict[str(q_results)].keys() and goldStandard_dict[str(q_results)][doc]==2):
                relevant_docs[doc]=tf_idf[doc]
            else:
                nrelevant_docs[doc]=tf_idf[doc]
    modified_query_tf_idf[str(q_results)]=rocchio(curAlpha,curBeta,curGamma,query_tf_idf[str(q_results)],relevant_docs,nrelevant_docs)
  
  # re-ranking documents based on query
    new_ranking = rankFunc(modified_query_tf_idf)

    # computing average value of NDCG@20 and MAP@20 for all the queries  
    avg_MAP = 0
    avg_NDCG = 0
    ndcg_na = 0

    for each_query in new_ranking:
        avg_MAP+=meanAP(each_query,new_ranking[each_query],20)
        ndcg_val = NDCG(each_query,new_ranking[each_query],20)
        # if NDCG function returns -1, then NDCG can't be calculated so we need to reduce the number of queries by 1 while average, else we can add NDCG 
        if ndcg_val>=0:
            avg_NDCG+=ndcg_val
        else:
            ndcg_na+=1

    # computing average values
    avg_MAP /= len(new_ranking) 
    avg_NDCG /= (len(new_ranking)-ndcg_na)
    final_results.append([curAlpha,curBeta,curGamma,avg_MAP,avg_NDCG])


# In[27]:


# creating file for storing result matrix
with open("Assignment3_13_rocchio_RF_metrics.csv",'w') as f:
    write = csv.writer(f)
    write.writerows(final_results)


# Pseudo-Relevant feedback

# In[28]:


# vector for storing final results
final_results = [["alpha","beta","gamma","mAP@20","NDCG@20"]]

# modifying query vector using rocchio for three sets of alpha, beta & gamma
for i in range(3):
    curAlpha = alpha[i]
    curBeta = beta[i]
    curGamma = gamma[i] 
    modified_query_tf_idf2={}
    # generating modified query using rocchio
    for q_results in range(1,len(top20_retrieved)+1):
        relevant_docs = {} # stores tf_idf of relevant docs in retrieved results for each query
        nrelevant_docs = {} # stores tf_idf of non-relevant docs in retrieved results for each query
        for doc in top20_retrieved[str(q_results)][:10]:
            relevant_docs[doc] = tf_idf[doc]
        modified_query_tf_idf2[str(q_results)] = rocchio(curAlpha,curBeta,curGamma,query_tf_idf[str(q_results)],relevant_docs,nrelevant_docs)
  
    # re-ranking documents based on query
    new_ranking = rankFunc(modified_query_tf_idf2) 

    # computing average value of NDCG@20 and MAP@20 for all the queries
    avg_MAP = 0
    avg_NDCG = 0
    ndcg_na = 0
    print(curAlpha, curBeta, curGamma)
    for each_query in new_ranking:
        avg_MAP+=meanAP(each_query,new_ranking[each_query],20)
        ndcg_val = NDCG(each_query,new_ranking[each_query],20)
        # if NDCG function returns -1, then NDCG can't be calculated so we need to reduce the number of queries by 1 while average, else we can add NDCG
        if ndcg_val>=0:
            avg_NDCG+=ndcg_val
        else:
            ndcg_na+=1
    
    # computing average values
    avg_MAP /= len(new_ranking)
    avg_NDCG /= (len(new_ranking)-ndcg_na)
    final_results.append([curAlpha,curBeta,curGamma,avg_MAP,avg_NDCG])


# In[30]:


# creating file for storing result matrix
with open("Assignment3_13_rocchio_PsRF_metrics.csv",'w') as f:
    write = csv.writer(f)
    write.writerows(final_results)


# In[ ]:




