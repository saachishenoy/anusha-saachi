# Overview 

This Quarter 1 Project attempts to recreate the CodeSearchNet project by combining keyword and semantic search techniques to create an advanced code search tool. Utilizing the CodeSearchNet Corpus, this project aims to significantly evaluate the NDCG and efficiency of querying source code across the Ruby programming language.

# Project Structure
The repository is organized as follows:
```
anusha-saachi-DSC180-Quarter1-Project/
├─ DiscussionWork/
│  ├─ BlockDiagram.pdf
│  ├─ q1-project-plan.md
├─ PreviousVersions/
│  ├─ CodeSearch Trial.ipynb
│  ├─ CodeSearchTrial.ipynb
│  ├─ CodeSearchV2.ipynb
│  ├─ CodeSearchV3.ipynb
│  ├─ CodeSearchV4.ipynb
├─ CodeSearch Final Version.ipynb
├─ annotationStore.csv
├─ code.py
├─ requirements.txt
├─ run.py
├─ README.md

```

# Usage

## Run Locally: 
Please Note: If you wish to run this locally, it will run, but it will take a VERY long time. We HIGHLY recommend you follow the "Run on DataHub" section instead.

1. Clone this repository on your local machine
2. Open your terminal
3. Change (cd) into the directory to the cloned repository
4. Make sure requirements.txt is installed. This contains all the necessary packages for running the project. Type  ``` pip install -r requirements.txt```
5. Use run.py to execute the search algorithm. It handles the integration of BM25 and semantic search methods. Type  ``` python run.py {insert query here}``` Where it says {insert query here}, replace this with the query you would like to run the search algorithm on.

## Run on DataHub
1. Download CodeSearch Final Version.ipynb
2. Upload this notebook to DataHub
3. Run every cell of this notebook
4. On the last cell, put your own query into the search function if you wish to test it out!

## Requirements
Python 3
Libraries listed in requirements.txt

## Exploring the Data
1. annotationStore.csv contains annotated data used to test our algorithm. This includes the labeled pairings and detailed annotations essential for understanding the test function rankings.

# Planning: 
The DiscussionWork/ directory contains our project plan and other discussion materials that guided our research.

# Contributors
1. Srianush Nandula
2. Saachi Shenoy

