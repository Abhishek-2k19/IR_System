#importing  all the libraries that are required
import nltk
import glob
import json
import csv
import sys
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer #initialising lemmatizer
lemmatizer = WordNetLemmatizer()

#creating the function to preprocess a given sentence and return a token of words
def preprocess(sentence):
    sentence = sentence.lower()          
    sentence = re.sub(r'[^\w\s]', '', sentence) #removing punctuations
    
    text_tokens = word_tokenize(sentence)
    tokens_without_sw = [lemmatizer.lemmatize(word) for word in text_tokens if not word in stopwords.words()] #lemmatizing and removing stop words
    
    return tokens_without_sw


#creating a dictionary to map paper_id with cord_id
reader = csv.DictReader(open('./Data/id_mapping.csv'))
id_mapping={}

for row in reader:
    id_mapping[row['paper_id']] = row['cord_id']
    

#reading all files of the cord-19 foleder recursively
files = glob.glob(sys.argv[1] + '/*', recursive = True) #path of dataset folder from the input

#dictionary to store the built inverted index
inverted_index={}


for file in files:
    f= open(file)
    data = json.load(f)
    sentence=""
    for obj in data['abstract']:
        sentence += obj['text'] +" "
    sentence = preprocess(sentence) #creating the sentence by looping over abstract and preprocessing

    for word in sentence: #for every word in sentence updating the inverted index
        if(word not in inverted_index):
            inverted_index[word]=[]
        if(word in inverted_index):
            if id_mapping[data['paper_id']] not in inverted_index[word]:
                inverted_index[word].append(id_mapping[data['paper_id']])

#sorting the postings of inverted index 
for lists in inverted_index:
    inverted_index[lists].sort()

#path to store the dictionary
pick_path = './model_queries_13.bin'

#convert the inverted index dictionary to pickle
with open (pick_path, 'wb') as pick:
    pickle.dump(inverted_index, pick)