import nltk
import numpy
import glob
import json
import csv
import pickle
import math
import itertools
import operator
import pandas as pd
import gc
import os
import sys

from nltk.tokenize import word_tokenize
nltk.download('punkt')
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('omw-1.4')
import re
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess(sentence):
    sentence = sentence.lower()          
    sentence = re.sub(r'[^\w\s]', '', sentence)
    
    text_tokens = word_tokenize(sentence)
    tokens_without_sw = [lemmatizer.lemmatize(word) for word in text_tokens if not word in stopwords.words()]
    
    return tokens_without_sw

reader = csv.DictReader(open('./Data/id_mapping.csv'))
id_mapping={}

for row in reader:
    id_mapping[row['paper_id']] = row['cord_id']
    
    
#define an empty dictionary
inverted_index ={}
pick_path = sys.argv[2]
# load the pickle contents
with open (pick_path,'rb') as pick:
    inverted_index.update(pickle.load(pick))
    
counter={}
for key in inverted_index:
    counter[key]= len(inverted_index[key])
sorted_df = dict( sorted(counter.items(), key=operator.itemgetter(1),reverse=True))
df = dict(itertools.islice(sorted_df.items(), 20000))

# reading dataset again to make the tf frequency
files = glob.glob(sys.argv[1]+'/*', recursive = True)
N= len(files)

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


reader = csv.DictReader(open('./Data/queries.csv'))
query={}
for row in reader:
    query[row['topic-id']] = row['query']

tf_query={}
for index in query:
    sentence = preprocess(query[index])
    tf_query[index]={}
    for word in sentence:
        if word in df:
            if word not in tf_query[index]:
                tf_query[index][word]=0
            tf_query[index][word]+=1
            
def normalize(df):
    for doc_id in df:
        sum=0;
        for w in df[doc_id]:
            sum+= df[doc_id][w]**2
        if(sum==0):
            continue
        for w in df[doc_id]:
            df[doc_id][w] /= math.sqrt(sum)

            
#for lnc.ltc scheme
#for df and idf of corpus
tf_idf={}
for doc_id in tf:
    tf_idf[doc_id]={}
    for word in tf[doc_id]:
        tf_idf[doc_id][word]= 1 + math.log(tf[doc_id][word],10)
normalize(tf_idf)
query_tf_idf={}
for index in tf_query:
    query_tf_idf[index]={}
    for word in tf_query[index]:
        query_tf_idf[index][word]= (1 + math.log(tf_query[index][word],10))* math.log(N/df[word])
normalize(query_tf_idf)
score={}
for q in query_tf_idf:
    score[q]={}
    for doc_id in tf_idf:
        score[q][doc_id]=0
        for word in query_tf_idf[q]:
            if word in tf_idf[doc_id]:
                score[q][doc_id]+= (tf_idf[doc_id][word] * query_tf_idf[q][word])

for q in query_tf_idf:
    score[q]=dict( sorted(score[q].items(), key=operator.itemgetter(1),reverse=True))
    score[q]=dict(itertools.islice(score[q].items(), 50))
csv=[]

for q in score:
    for doc in score[q]:
        csv.append([q,doc]);
df2 = pd.DataFrame(csv, columns=['query_id', 'document_id'])
df2.to_csv("Assignment2_13_ranked_list_A.csv",index=False)

#for lnc.lpc scheme
#for df and idf of corpus
tf_idf={}
for doc_id in tf:
    tf_idf[doc_id]={}
    for word in tf[doc_id]:
        tf_idf[doc_id][word]= 1 + math.log(tf[doc_id][word],10)
normalize(tf_idf)
query_tf_idf={}
for index in tf_query:
    query_tf_idf[index]={}
    for word in tf_query[index]:
        query_tf_idf[index][word]= (1 + math.log(tf_query[index][word],10))* max(0,math.log((N-df[word])/df[word]))
                                                                                     
normalize(query_tf_idf)

score={}

for q in query_tf_idf:
    score[q]={}
    for doc_id in tf_idf:
        score[q][doc_id]=0
        for word in query_tf_idf[q]:
            if word in tf_idf[doc_id]:
                score[q][doc_id]+= (tf_idf[doc_id][word] * query_tf_idf[q][word])

for q in query_tf_idf:
    score[q]=dict( sorted(score[q].items(), key=operator.itemgetter(1),reverse=True))
    score[q]=dict(itertools.islice(score[q].items(), 50))
    
csv=[]

for q in score:
    for doc in score[q]:
        csv.append([q,doc]);
df2 = pd.DataFrame(csv, columns=['query_id', 'document_id'])
df2.to_csv("Assignment2_13_ranked_list_B.csv",index=False)

#for anc.apc scheme
#for df and idf of corpus
tf_idf={}
for doc_id in tf:
    tf_idf[doc_id]={}
    
    mx =0
    for word in tf[doc_id]:
        mx= max(mx,tf[doc_id][word])
    for word in tf[doc_id]:
        if mx==0:
            tf_idf[doc_id][word]=0
        else:
            tf_idf[doc_id][word]= 0.5 + ((0.5*tf[doc_id][word])/(mx))
normalize(tf_idf)
query_tf_idf={}
for index in tf_query:
    query_tf_idf[index]={}
    mx =0
    for word in tf_query[index]:
        mx=max(mx, tf_query[index][word])
    for word in tf_query[index]:
        if mx==0:
            query_tf_idf[index][word]=0
        else:
            query_tf_idf[index][word]= (0.5 + ((0.5 + tf_query[index][word])/(mx)))* max(0,math.log((N-df[word])/df[word]))
                                                                                     
normalize(query_tf_idf)

score={}

for q in query_tf_idf:
    score[q]={}
    for doc_id in tf_idf:
        score[q][doc_id]=0
        for word in query_tf_idf[q]:
            if word in tf_idf[doc_id]:
                score[q][doc_id]+= (tf_idf[doc_id][word] * query_tf_idf[q][word])

for q in query_tf_idf:
    score[q]=dict( sorted(score[q].items(), key=operator.itemgetter(1),reverse=True))
    score[q]=dict(itertools.islice(score[q].items(), 50))
csv=[]

for q in score:
    for doc in score[q]:
        csv.append([q,doc]);
df2 = pd.DataFrame(csv, columns=['query_id', 'document_id'])
df2.to_csv("Assignment2_13_ranked_list_C.csv",index=False)
