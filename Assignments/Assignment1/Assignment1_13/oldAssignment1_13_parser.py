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

# path = "queries.csv"
path = sys.argv[1]

def preprocess(sentence):
    sentence = sentence.lower()          
    sentence = re.sub(r'[^\w\s]', '', sentence) #removing punctuations
    
    text_tokens = word_tokenize(sentence)
    tokens_without_sw = [lemmatizer.lemmatize(word) for word in text_tokens if not word in stopwords.words()] #lemmatizing and removing stop words
    
    return tokens_without_sw

outputFile = open("queries_13.txt",'w')
with open(path,mode='r') as file:
  queryFile = csv.reader(file)

  for queryNum, queryLine in enumerate(queryFile):
    if queryNum==0: continue
    processedQuery = preprocess(queryLine[1])
    outputFile.write(str(queryNum)+","+' '.join(processedQuery)+"\n")

outputFile.close()

