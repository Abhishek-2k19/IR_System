#importing  all the libraries that are required
import csv
import sys
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer #initialising lemmatizer
lemmatizer = WordNetLemmatizer()

# Reading the path of raw queries file
path = sys.argv[1]


# preprocessing function for removing stop words and punctuation marks and for performing lemmatization (without POS tags) to generate tokens
def preprocess(sentence):
    sentence = sentence.lower()          
    sentence = re.sub(r'[^\w\s]', '', sentence) #removing punctuations
    
    text_tokens = word_tokenize(sentence)
    tokens_without_sw = [lemmatizer.lemmatize(word) for word in text_tokens if not word in stopwords.words()] #lemmatizing and removing stop words
    
    return tokens_without_sw

# creating/overwriting file for storing the processed queries
outputFile = open("queries_13.txt",'w')

# opening the raw query file in read mode
with open(path,mode='r') as file:
  queryFile = csv.reader(file) # reading the csv file
	
  # iterating on each query and processing it and writing the processed query to outputFile
  for queryNum, queryLine in enumerate(queryFile):
    if queryNum==0: continue # skipping Table headings 
    processedQuery = preprocess(queryLine[1])
    outputFile.write(str(queryNum)+","+' '.join(processedQuery)+"\n")

# closing the outputFile 
outputFile.close()

