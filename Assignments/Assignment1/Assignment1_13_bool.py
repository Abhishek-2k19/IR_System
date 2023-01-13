import pickle
import sys
import functools


inverted_index={}
pick_path = sys.argv[1]#'./model_queries_13.bin'

with open (pick_path,'rb') as pick:
    inverted_index.update(pickle.load(pick))

def merge(index1, index2):
    res = []
    i=0
    j=0
    while i<len(index1) and j<len(index2):
        if index1[i]==index2[j]:
            res.append(index1[i])
            i=i+1
            j=j+1
        elif index1[i]<index2[j]:
            i=i+1
        else:
            j=j+1
    return res

def cmp(x,y):
    return len(inverted_index[x])-len(inverted_index[y])

outputFile = open("Assignment1_13_results.txt",'w')
path= sys.argv[2]#'./queries_13.txt'
queries_file = open(path, "r")
data = queries_file.read()
data_into_list = data.split("\n")

for query_list in data_into_list: 
    query = query_list.split(",")
    if len(query)==2:     
        tokens_list = query[1].split(" ")
        token_list = sorted(tokens_list,key=functools.cmp_to_key(cmp))
        i = 0
        for tokens in token_list:
            if i == 0:
                answer=inverted_index[tokens]
            else:
                answer=merge(answer,inverted_index[tokens])
            i= i+1
        outputFile.write(query[0]+":"+str(answer)+"\n"+"\n")

        


outputFile.close()
queries_file.close()