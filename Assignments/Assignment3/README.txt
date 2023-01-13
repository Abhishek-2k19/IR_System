GROUP 13

Group members - 
Abhishek S Purohit - 19CH10002 
Yash Jain - 19CH10078
Sanskar Patni - 19CH10046

python version==3.10.7 

specific library requirements
nltk==3.6.7
numpy==1.19.5

Apart from above mentioned libraries, our code uses various other libraries like glob, json, csv, sys, pickle, os, sys and few others which are already part of python's standard library and need not require any separate installation. Apart from that, our code downloads the following components of nltk (if not found in system) 
1. punkt
2. stopwords
3. omw-1.4
4. wordnet

Following commands (present in our code) downloads the above components. (It could be separately used to download these components through Python Interpreter)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

our code will create few temporary files in "./temp_tf/" directory which will be used for internal processing. It could be safely deleted once the program exits. Also while running the program, kindly ensure that there is no such directory with same name or if its there, then make sure that it does NOT contain any files.   


