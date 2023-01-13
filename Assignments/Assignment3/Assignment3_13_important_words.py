#!/usr/bin/env python
# coding: utf-8

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
path_to_listA = sys.argv[3]


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

# generating tf queries in the same way as we did for corpus    
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


# generating tf_idf
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


# In[14]:


# function to merge two dictionaries (adding the values corresponding to the same keys)
def Merge(dict1, dict2):
    temp=dict2.copy()
    for key in dict1:
        if(key not in temp):
            temp[key]=0
        temp[key]+=dict1[key]
      
    return temp

# function to multiply the dictionary with a constant number
def prod(Dict,val):
    temp = Dict.copy()
    for key in temp:
        temp[key]=val*temp[key]
    return temp


# In[15]:


# reading the retrieved rank list
rankedListFile = open(path_to_listA,'r')
rankedList = csv.reader(rankedListFile)
# top20_retrieved = {} # top20_retrieved[query-id] = [list of top 20 doc id], everything in string, including query-id 

# creating a dictionary to store the tf_idf of words averaged across top 10 queries
average_tf_idf = {}
for itr,result in enumerate(rankedList,0):
    # taking only top 10 documents
    if(itr%50 >=1 and itr%50 <=10):
        # if its first document, then we need to initialize average_tf_idf[query-id] with empty dictionary
        if(itr%50==1):
            average_tf_idf[result[0]] = {}
        # merging the tf_idf[current_doc] with the average tf_idf
        average_tf_idf[result[0]] = Merge(average_tf_idf[result[0]],tf_idf[result[1]])

# closign the rankedListFile
rankedListFile.close()


# In[17]:


# averaging the tf_idf_sum with number of docs (10)
for each_query in average_tf_idf:
    average_tf_idf[each_query] = prod(average_tf_idf[each_query],1/10)


# In[19]:


# storing the final result to list
final_results = []
final_results.append(["query-id","important-words"])
for query in average_tf_idf:
    # sorting the words according to its score and taking top 5 important words
    temp1 = dict( sorted(average_tf_idf[query].items(), key=operator.itemgetter(1),reverse=True))
    current_result = [query]
    important_words = ""
    for itr,word in enumerate(temp1,1):
        if itr==1:
            important_words = word
        else:
            important_words = important_words +","+word
        # breaking after taking top 5 words
        if itr==5:
            break
    current_result.append(important_words)
    final_results.append(current_result)


# In[21]:


# saving the file to the disk 
with open("Assignment3_13_important_words.csv","w") as f:
    write = csv.writer(f)
    write.writerows(final_results)


# In[ ]:




