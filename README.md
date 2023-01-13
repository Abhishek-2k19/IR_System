# IR_System
Information retrieval system for retrieving the relevant research paper for COVID related query from the COVID research paper dataset.

# Assignment_folder
This contains the assignment pdf along with the submission(code) for each assignment organized in the folder.


# Dataset
Dataset folder contains the covid related research paper dataset and directory is arranged as follows:

[Main Code Directory]<br>
|<br>
|-------- Assignment_folder<br>
|<br>
|-------- Data (Not present in the repo, download it from the link given below and rename the folder to "Data" after extracting)<br> 
| |<br>
| |---------- CORD-19 (Ensure that inside this folder, there are all the json files)<br>
| |<br>
| |---------- queries.csv<br>
| |<br>
| |---------- id_mapping.csv<br>
| |<br>
| |---------- qrels.csv<br>


"Data" can be downloaded from <a href="https://drive.google.com/file/d/1UGnm4v7ZTNWaoYOsTZ61DKol6eEk8Aa2/view?usp=sharing">here</a>.

User needs to copy the desired code files to the main code directory, for example, user needs to build the IR system as given in Assignment_folder/Assignment1.pdf , then it needs to copy the files from Assignment_folder/Assignment1/* to main code directory and then run the desired script to generate the results in the same format as specified in the corresponding assignment pdf file. 

Since the dataset is large enough, it will take some time to run the script, in case, you cannot generate the TF-IDF file for Assignment 3, then you can download the same from <a href="">here</a>. 

# About Project
This is the term project for the Information retrieval course-CS60092 at IIT Kharagpur.

This IR system is jointly developed by:
1. Abhishek S Purohit
2. Sanskar Patni
3. Yash Jain

