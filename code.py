#!/usr/bin/env python
# coding: utf-8

# In[1]:


def get_data():
    dataset = load_dataset("code_search_net", "ruby")
    return dataset


# In[2]:


def encode_semantic(dataset):
    documents = dataset['train']['func_documentation_string']
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    encoded_data = model.encode(documents)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(model.get_sentence_embedding_dimension()))
    index.add_with_ids(encoded_data, np.array(range(0, len(documents))))
    faiss.write_index(index, 'sample_documents')
    index = faiss.read_index('sample_documents')
    return model, index


# In[3]:


def semantic_search(query, model, index):
    t = time.time()
    query_vector = model.encode([query])
    
    top_k = index.search(query_vector, index.ntotal)
    
    document_ids = top_k[1].tolist()[0]
    scores = top_k[0].tolist()[0]
    
    results = [(doc_id, score) for doc_id, score in zip(document_ids, scores)]
    
    results.sort(key=lambda x: x[0])
    
    semantic_scores = [i[1] for i in results]

    max_score = max(semantic_scores)
    normalized_semantic_scores = [score / max_score for score in semantic_scores]
    normalized_semantic_scores
    
    return normalized_semantic_scores


# In[4]:


def encode_BM25(dataset):
    func_tokens = dataset['train']['func_code_tokens']
    bm25 = BM25Okapi(func_tokens)
    return bm25


# In[5]:


def bm25_search(user_input, bm25):
    
    doc_scores = bm25.get_scores(user_input)
    max_score = max(doc_scores)

    # normalize BM25 scores
    normalized_doc_scores = [score / max_score for score in doc_scores]
    
    return normalized_doc_scores


# In[6]:


def find_quartiles(data):
    filtered_sorted_data = sorted([x for x in data if x != 0])
    n = len(filtered_sorted_data)
    if n == 0:
        # Handle the case where all values are zero
        return [0 for _ in data]
    q1 = filtered_sorted_data[int(n * 0.25) - 1]
    q2 = filtered_sorted_data[int(n * 0.5) - 1]
    q3 = filtered_sorted_data[int(n * 0.75) - 1]

    quartiles = []
    for value in data:
        if value <= q1:
            quartiles.append(0)
        elif value <= q2:
            quartiles.append(1)
        elif value <= q3:
            quartiles.append(2)
        else:
            quartiles.append(3)
    return quartiles


# In[16]:


def search_results(sem_weight, bm_weight, user_input):
    
    sem = semantic_search(user_input, model, index)
    bm = bm25_search(user_input, bm25)
    weighted_sem = [i * sem_weight for i in sem]
    weighted_bm = [i * bm_weight for i in bm]
    weighted_avg = [weighted_sem[i]+ weighted_bm[i] for i in range(0, len(weighted_bm))]
    sum_weight = sem_weight + bm_weight 
    weighted_avg_norm = [i/sum_weight for i in weighted_avg]
    url = dataset["train"]['func_code_url']
    if not weighted_avg_norm or np.isnan(weighted_avg_norm).any():
        return {} 
    
    try:
        labels = find_quartiles(weighted_avg_norm)
        output_dict = {url[i]: labels[i] for i in range(len(weighted_avg_norm))}
        
    except ValueError:
        return {}
    
    return output_dict


# In[8]:


def ndcg(search_output, eval_dict):
    inter = list(search_output.keys() & eval_dict.keys())
    y_pred = [search_output[i] for i in search_output.keys() if i in inter]
    y_true = [eval_dict[i] for i in eval_dict.keys() if i in inter]
    
    if (len(y_true) <=1):
        return None
    
    if (len(y_pred) <=1):
        return None
    
    from sklearn.metrics import ndcg_score
        
    y_true_nd = np.zeros(shape=(len(y_true), 4))
    y_true_nd[np.arange(len( y_true)), y_true] = 1
    y_pred_nd = np.zeros(shape=(len(y_true), 4))
    y_pred_nd[np.arange(len( y_true)), y_pred] = 1
    return ndcg_score(y_true_nd, y_pred_nd)


# In[13]:


def find_best_weights(sem_weight, bm_weight):

    grouped_evals = eval_Ruby.groupby('Query').apply(lambda x: pd.Series(x.Relevance.values, index=x.GitHubUrl).to_dict())

    ndcgs = []
    search_results_cache = {}  

    for query, evals in grouped_evals.items():
        if (sem_weight, bm_weight, query) not in search_results_cache:
            search_results_cache[(sem_weight, bm_weight, query)] = search_results(sem_weight, bm_weight, query)
        
        our_search = search_results_cache[(sem_weight, bm_weight, query)]

        ndcgs.append(ndcg(our_search, grouped_evals[query]))
    filtered_scores = [s for s in ndcgs if s is not None]
    return  sum(filtered_scores)/len(filtered_scores)


# In[17]:


def grid_search():
    best_precision = 0
    best_weights = (0, 0)
    sem_weight_range = (0, 1) 
    increment = 0.1
    

    for sem_weight in np.arange(*sem_weight_range, increment):
        bm_weight = 1 - sem_weight
        current_precision = find_best_weights(sem_weight, bm_weight)
        if current_precision > best_precision:
            best_precision = current_precision
            best_weights = (sem_weight, bm_weight)

    return best_weights


# In[20]:


def end_to_end(best_weights):
    grouped_evals = eval_Ruby.groupby('Query').apply(lambda x: pd.Series(x.Relevance.values, index=x.GitHubUrl).to_dict())

    prec = []
    search_results_cache = {}  

    sem_weight, bm_weight = best_weights
    
    for query, evals in grouped_evals.items():
        if (sem_weight, bm_weight, query) not in search_results_cache:
            search_results_cache[(sem_weight, bm_weight, query)] = search_results(sem_weight, bm_weight, query)
        
        our_search = search_results_cache[(sem_weight, bm_weight, query)]
        inter = set(evals.keys()) & set(our_search.keys())

        if inter:
            prec.append(ndcg(our_search, evals))
        precs = [i for i in prec if i is not None]
    return sum(precs) / len(precs) if precs else 0


# In[26]:


def search(best_weights, user_input):
    sem_weight, bm_weight = best_weights
    sem = semantic_search(user_input, model, index)
    bm = bm25_search(user_input, bm25)
    weighted_sem = [i * sem_weight for i in sem]
    weighted_bm = [i * bm_weight for i in bm]
    weighted_avg = [weighted_sem[i]+ weighted_bm[i] for i in range(0, len(weighted_bm))]
    sum_weight = sem_weight + bm_weight 
    weighted_avg_norm = [i/sum_weight for i in weighted_avg]
    url = []
    function_name = []
    doc_string = []
    title =[]
    top_10_docs = sorted(range(len(weighted_avg_norm)), key=lambda i: weighted_avg_norm[i], reverse=True)[:10]    
    for i in top_10_docs:
        function_name.append(dataset['train']['func_name'][i])
        doc_string.append(dataset['train']['func_documentation_string'][i])
        url.append(dataset["train"]['func_code_url'][i])
        
    results_df = pd.DataFrame({'URL': url})
    return results_df


# In[ ]:




