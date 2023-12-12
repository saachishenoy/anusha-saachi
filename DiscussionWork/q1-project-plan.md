Architecture: what steps will search do?
1. Import/ Train
  Import the dataset.
  Import the 'BAAI/bge-base-en-v1.5' sentence transformer model and tune the transformer with our data.
2. Inverted Index
  Create an inverted index that includes both unigrams and bigrams. This is used to locate documents quickly in the TF-IDF search.
3. Query Entry
  User input is processed as the query.
4. Semantic Search
  The query is encoded with the sentence transformer into a vector and that vector is used to find similar documents. The top 100 similar documents are outputted.
5. TF-IDF Search
  The query is tokenized and there is a TF-IDF search on each distinct token and a search on bigram token combinations. The TF-IDF results are summed per document. The top 100 similar documents are outputted.
6. Merge Search
  The final result is the intersection between the top results from both semantic and TF-IDF searches. However, if there are not enough points in the intersection to produce 10 query results, then the search fills the rest of the ranking with the top n not intersecting semantic results.




Algorithms: What is the math inside the steps in the architecture?
	For the TF-IDF Search, unigrams and bigrams are generated from the code tokens. After that, an inverted index is made where each token has a corresponding list of documents that contain that specific unigram / bigram. To find the TF-IDF of each token, the TF and IDF metrics are multiplied. TF is term frequency, or how many times the term is found in a particular document. IDF is the inverse document frequency, which tells us how common or rare the unigram/bigram is among all documents. When searching for a certain phrase, documents are ranked by the sum of their TF-IDF scores for all the words in that phrase.
	In semantic search, the main calculations occur when the trained SentenceTransformer model encodes the documentation string into embeddings. These embeddings are added to a FAISS index for a vector similarity search. For a given query, semantic search is then able to find the top-k most similar documents.
	Our final search function involved a bit of math to combine these two methods. First we extracted the top 100 results from each type of search and found the overlapping documents among these results. Then, we used the overlapping documents as the top search results. If there were less than 10 overlapping documents, we retrieved the rest of the results from the top 100 documents using semantic search since it has proven to be more accurate. 



Dataset: which portions of the dataset will you use?
	The dataset is the ‘ruby’ subsection of the greater CodeSearchNet dataset. Specifically, the semantic search is tuned and run on the untokenized doc string. This is the column labeled func_documentation_string. The inverted index and TF-IDF search were created and are run on the tokenized code itself. This column is labeled func_code_tokens. The results include the document id, function name, and function docstring for all 10 query results.



Evaluation: how will you measure the results? Do you have benchmarks?
	As of now, our evaluation methods have involved manually evaluating the results of queries and giving them our own relevance metric. However, we are planning to include more concrete evaluation metrics such as precision and recall. For further evaluation, we plan to use the NDCG metric, which finds the relevance of the documents returned for each query. We'll create benchmark queries and manually gather the associated documents. Then, we’ll search these queries with our search, and then we can calculate the NDCG score for each. This shows the quality of the rankings in comparison to our manually gathered benchmark. 



Improvements: List of future work / stretch goals that we would like to work toward.
1. Implement relevance feedback mechanisms based on user experience
  a. If a user searches up a certain query and interacts with particular documents, it would be useful to keep a record of this for the next time a user searches up the same query
  b. The documents with most interactions from that search should show up at the top
2. Currently we are only using functions in the programing language Ruby since it had the smallest dataset, we want to expand our search to all programming languages so that it would be useful for users using any language
  a. To do so, we want to make our code more efficient so it can run faster on larger databases
3. We want to include more preprocessing steps for semantic search to improve the quality and efficiency of this part of our algorithm
  a. Since the data we are using for TF-IDF is already tokenized there is less preprocessing we can do with that information.
