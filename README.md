# anusha-saachi

To run via Terminal
1) Clone this repository on your local machine
2) Open terminal
3) Change directory to the cloned repository
4) Run $ pip install -r requirements.txt
5) Run $ python3 run.py <query>




To run via JupyterHub

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

## To Run Locally:

1. Clone this repository on your local machine
2. Open your terminal
3. Change directory to the cloned repository
4. Make sure requirements.txt is installed. This contains all the necessary packages for running the project. Type  ``` pip install -r requirements.txt```
5. Use run.py to execute the search algorithm. It handles the integration of BM25 and semantic search methods. Type  ``` python run.py {insert query here}``` Where it says {insert query here}, replace this with the query you would like to run the search algorhithim on.

## To run on DataHub
1. Download CodeSearch Final Version.ipynb
2. Upload this notebook to DataHub
3. Run every cell of this notebook
4. On the last cell put your own query into the search function if you wish to test it out!


## Exploring the Data
1. annotationStore.csv contains annotated data used to test our algorithm. This includes the labeled pairings and detailed annotations essential for understanding the test function rankings.


# Planning: 
The DiscussionWork/ directory contains our project plan and other discussion materials that guided our research.

Requirements
Python 3.x
Libraries as listed in requirements.txt
Contributing
To contribute to this project, please fork the repository and submit a pull request with your proposed changes.

Contact
For any inquiries or contributions, feel free to contact us through the repository's issue tracker.

Acknowledgments
We extend our gratitude to the CodeSearchNet Corpus for providing a comprehensive dataset that played a pivotal role in our research.
