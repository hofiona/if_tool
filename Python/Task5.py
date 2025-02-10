"""
Task 5 – Use three effectiveness measures to evaluate the three models.
"""

import os
import math
import re
import string
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from collections import Counter

stemmer = PorterStemmer()

# Fetch stop words from the file
def get_stop_words(filepath):
    """
    Fetch stop words from the file.

    Parameters:
    filepath (str): Path to the stop words file.

    Returns:
    list: List of stop words.
    """
    with open(filepath) as file:
        content = file.readline().strip().split(',')
    return content

stop_words = get_stop_words("common-english-words.txt")
queryfile = "the50Queries.txt"
outputFolder = "RankingOutputs"
data_collection_folder = "Data_Collection"
evaluation_benchmark_folder = "EvaluationBenchmark"

class DataDoc:
    def __init__(self, docID):
        """
        Initialize a DataDoc object.

        Parameters:
        docID (str): Document ID.
        """
        self.docID = docID
        self.terms = dict()
        self.doc_len = 0

    def set_doc_len(self, count):
        """
        Set the document length.

        Parameters:
        count (int): Document length.
        """
        self.doc_len = count

    def getDocId(self):
        """
        Get the document ID.

        Returns:
        str: Document ID.
        """
        return self.docID

    def getDocLen(self):
        """
        Get the document length.

        Returns:
        int: Document length.
        """
        return self.doc_len

    def get_term_list(self):
        """
        Get the term list.

        Returns:
        dict: Dictionary of terms and their frequencies.
        """
        return self.terms

    def add_term(self, term):
        """
        Add a term to the document.

        Parameters:
        term (str): Term to add.
        """
        if term not in self.terms:
            self.terms[term] = 1
        else:
            self.terms[term] += 1

class DataColl:
    def __init__(self):
        """
        Initialize a DataColl object.
        """
        self.collections = {}
        self.numOfDocs = 0
        self.totalDocLength = 0
        self.avgDocLen = 0

    def add_doc(self, doc):
        """
        Add a document to the collection.

        Parameters:
        doc (DataDoc): Document to add.
        """
        self.collections[doc.getDocId()] = doc
        self.totalDocLength += doc.getDocLen()
        self.numOfDocs += 1

    def get_coll(self):
        """
        Get the collection of documents.

        Returns:
        dict: Dictionary of documents.
        """
        return self.collections

    def getTotalDocLen(self):
        """
        Get the total document length.

        Returns:
        int: Total document length.
        """
        return self.totalDocLength

    def getNumOfDocs(self):
        """
        Get the number of documents.

        Returns:
        int: Number of documents.
        """
        return self.numOfDocs

    def setAvgDocLen(self, avg):
        """
        Set the average document length.

        Parameters:
        avg (float): Average document length.
        """
        self.avgDocLen = avg

    def getAvgDocLen(self):
        """
        Get the average document length.

        Returns:
        float: Average document length.
        """
        return self.avgDocLen

def my_df(coll):
    """
    Calculate document frequencies for terms.

    Parameters:
    coll (dict): Collection of documents.

    Returns:
    dict: Dictionary of term frequencies.
    """
    docFreq = {}
    for _, doc in coll.items():
        terms = doc.get_term_list()
        for term in terms:
            docFreq[term] = docFreq.get(term, 0) + 1
    return docFreq

def avg_length(coll):
    """
    Calculate the average document length.

    Parameters:
    coll (DataColl): Collection of documents.

    Returns:
    float: Average document length.
    """
    avg_length = coll.getTotalDocLen() / coll.getNumOfDocs()
    coll.setAvgDocLen(avg_length)
    return avg_length

def get_re_docs(query_id):
    """
    Get relevant documents for a query.

    Parameters:
    query_id (str): Query ID.

    Returns:
    dict: Dictionary of relevant documents and their relevance scores.
    int: Total number of relevant documents.
    """
    relevance_info = {}
    filename = f"Dataset{query_id[-3:]}.txt"
    filepath = os.path.join(evaluation_benchmark_folder, filename)
    total_relevant_docs = 0
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3 and parts[0] == query_id:
                    doc_id = parts[1]
                    relevance_score = int(parts[2])
                    relevance_info[doc_id] = relevance_score
                    if relevance_score == 1:
                        total_relevant_docs += 1
    return relevance_info, total_relevant_docs

def parse_query(query):
    """
    Parse a query.

    Parameters:
    query (str): Query string.

    Returns:
    dict: Dictionary of parsed query terms and their frequencies.
    """
    curr_word = dict()
    query = query.translate(str.maketrans('', '', string.digits)).translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    query = re.sub(r"\s+", " ", query)
    for term in query.split():
        original_term = term.lower()
        term = stemmer.stem(original_term)
        if len(term) > 2 and term not in stop_words:
            curr_word[term] = curr_word.get(term, 0) + 1

    return curr_word

def load_queries(query_file):
    """
    Load queries from a file.

    Parameters:
    query_file (str): Path to the query file.

    Returns:
    dict: Dictionary of queries.
    """
    queries = {}
    try:
        with open(query_file, 'r', encoding='utf-8') as file:
            query_data = file.read()
            query_pattern = re.compile(
                r'<Query>\s*'
                r'<num> Number: (R\d+)\s*'
                r'<title> (.*?)\s*'
                r'<desc> Description:\s*(.*?)\s*'
                r'<narr> Narrative:\s*(.*?)\s*'
                r'</Query>', re.DOTALL
            )
            matches = query_pattern.findall(query_data)
            for query_id, title, description, narrative in matches:
                title_query = f"{title}"
                parsed_query = parse_query(title_query)
                queries[query_id.strip()] = parsed_query
    except Exception as e:
        print(f"Failed to load queries from {query_file}: {e}")
    return queries

def bm25(coll, query_id, query_terms, df):
    """
    Calculate BM25 scores for a query.

    Parameters:
    coll (DataColl): Collection of documents.
    query_id (str): Query ID.
    query_terms (dict): Query terms and their frequencies.
    df (dict): Document frequencies for terms.

    Returns:
    dict: Dictionary of BM25 scores for documents.
    """
    scores = {}
    k1 = 1.2
    k2 = 500
    b = 0.75
    N = coll.getNumOfDocs()
    avg_docLen = coll.getAvgDocLen()

    for docId, doc in coll.get_coll().items():
        termFreq = doc.get_term_list()
        scores[docId] = 0
        dl_avdl = doc.getDocLen() / avg_docLen
        K = k1 * ((1 - b) + (b * dl_avdl))

        for word_i, qfi in query_terms.items():
            ni = df.get(word_i, 0)
            fi = termFreq.get(word_i, 0)
            R = 0
            ri = 0
            num1 = (ri + 0.5) / (R - ri + 0.5)
            num2 = (ni - ri + 0.5) / (N * 2 - ni - R + ri + 0.5)
            k1_attr = ((k1 + 1) * fi) / (K + fi)
            k2_attr = ((k2 + 1) * qfi) / (k2 + qfi)
            scores[docId] += (math.log10(num1 / num2) * k1_attr * k2_attr)

    return scores

def jm_smoothing(query, doc_term_freq, doc_length, collection_term_freq, collection_length, lambda_param):
    """
    Perform Jelinek-Mercer smoothing for a query.

    Parameters:
    query (dict): Query terms and their frequencies.
    doc_term_freq (dict): Term frequencies in the document.
    doc_length (int): Length of the document.
    collection_term_freq (dict): Term frequencies in the collection.
    collection_length (int): Total length of the collection.
    lambda_param (float): Smoothing parameter.

    Returns:
    float: Smoothed probability.
    """
    product = 1.0
    for term in query:
        doc_freq = doc_term_freq.get(term, 0)
        coll_freq = collection_term_freq.get(term, 0)
        doc_prob = doc_freq  / doc_length if doc_length > 0 else 0
        coll_prob = coll_freq / collection_length if collection_length > 0 else 0
        smoothed_prob = ((1 - lambda_param) * doc_prob) + (lambda_param * coll_prob)
        product *= smoothed_prob if smoothed_prob > 0 else 1
    return product

def rank_documents_jm(query_id, query, documents, collection_term_freq, collection_length):
    """
    Rank documents using Jelinek-Mercer smoothing.

    Parameters:
    query_id (str): Query ID.
    query (dict): Query terms and their frequencies.
    documents (dict): Collection of documents.
    collection_term_freq (dict): Term frequencies in the collection.
    collection_length (int): Total length of the collection.

    Returns:
    list: List of ranked documents and their scores.
    """
    scores = []
    for doc_id, doc in documents.items():
        doc_term_freq = doc.get_term_list()
        doc_length = doc.getDocLen()
        score = jm_smoothing(query, doc_term_freq, doc_length, collection_term_freq, collection_length, 0.4)
        scores.append((doc_id, score))
    return sorted(scores, key=lambda item: item[1], reverse=True)

def parse_collection(inputpath):
    """
    Parse documents in a collection.

    Parameters:
    inputpath (str): Path to the collection folder.

    Returns:
    DataColl: DataColl object containing parsed documents.
    """
    coll = DataColl()
    files = os.listdir(inputpath)
    for file in files:
        file_path = os.path.join(inputpath, file)
        if not file.endswith(".xml"):
            continue  # Skip non-XML files
        start_end = False
        word_count = 0

        with open(file_path, 'r', encoding='utf-8') as myfile:
            lines = myfile.readlines()
            for line in lines:
                line = line.strip()
                if not start_end:
                    if line.startswith("<newsitem "):
                        for part in line.split():
                            if part.startswith("itemid="):
                                dataDoc = DataDoc(part.split("=")[1].split("\"")[1])
                                print(f"Processing document with ID: {dataDoc.getDocId()}")
                                break
                    if line.startswith("<text>"):
                        start_end = True
                elif line.startswith("</text>"):
                    break
                else:
                    line = line.replace("<p>", "").replace("</p>", "")
                    line = line.translate(str.maketrans('', '', string.digits)).translate(
                        str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                    line = re.sub(r"\s+", " ", line)
                    for term in line.split():
                        stemmed_term = stemmer.stem(term.lower())
                        word_count += 1
                        if len(stemmed_term) > 2 and stemmed_term not in stop_words:
                            dataDoc.add_term(stemmed_term)
        dataDoc.set_doc_len(word_count)
        coll.add_doc(dataDoc)
    return coll

def parse_query_long(query):
    """
    Parse a long query.

    Parameters:
    query (str): Query string.

    Returns:
    dict: Dictionary of parsed query terms and their frequencies.
    """
    curr_word = dict()
    query = query.translate(str.maketrans('', '', string.digits)).translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    query = re.sub(r"\s+", " ", query)
    for term in query.split():
        original_term = term.lower()
        term = stemmer.stem(original_term)
        if len(term) > 2 and term not in stop_words:
            curr_word[term] = curr_word.get(term, 0) + 1

    return curr_word

def load_queries_long(query_file):
    """
    Load long queries from a file.

    Parameters:
    query_file (str): Path to the query file.

    Returns:
    dict: Dictionary of long queries.
    """
    queries = {}
    try:
        with open(query_file, 'r', encoding='utf-8') as file:
            query_data = file.read()
            query_pattern = re.compile(
                r'<Query>\s*'
                r'<num> Number: (R\d+)\s*'
                r'<title> (.*?)\s*'
                r'<desc> Description:\s*(.*?)\s*'
                r'<narr> Narrative:\s*(.*?)\s*'
                r'</Query>', re.DOTALL
            )
            matches = query_pattern.findall(query_data)
            for query_id, title, description, narrative in matches:
                full_query_content = f"{title} {description} {narrative}"
                parsed_query = parse_query_long(full_query_content)
                queries[query_id.strip()] = parsed_query
    except Exception as e:
        print(f"Failed to load queries from {query_file}: {e}")
    return queries

def probabilistic_relevance_feedback(query_terms, relevant_docs, non_relevant_docs, df, N, alpha=10, beta=5, l1_lambda=0.99):
    """
    Perform probabilistic relevance feedback.

    Parameters:
    query_terms (dict): Query terms and their frequencies.
    relevant_docs (dict): Relevant documents and their term frequencies.
    non_relevant_docs (dict): Non-relevant documents and their term frequencies.
    df (dict): Document frequencies for terms.
    N (int): Total number of documents.
    alpha (float): Weight for relevant terms.
    beta (float): Weight for new terms from relevant documents.
    l1_lambda (float): Regularization parameter.

    Returns:
    dict: Updated query terms and their weights.
    """
    epsilon = 1e-11  # Small constant to avoid math domain errors
    new_query_terms = Counter(query_terms)
    term_weights = {}  # To store the calculated term weights for L1 regularization

    # Calculate the probabilities for relevant documents
    for term in query_terms:
        p_r = sum(relevant_docs[doc].get(term, 0) for doc in relevant_docs) / len(relevant_docs) if relevant_docs else 0
        p_nr = sum(non_relevant_docs[doc].get(term, 0) for doc in non_relevant_docs) / len(
            non_relevant_docs) if non_relevant_docs else 0

        # Ensure probabilities are within valid range
        p_r = max(min(p_r, 1 - epsilon), epsilon)
        p_nr = max(min(p_nr, 1 - epsilon), epsilon)

        new_weight = math.log((p_r * 4 * (1 - p_nr)) / ((1 - p_r) * p_nr))
        new_query_terms[term] += alpha * new_weight
        term_weights[term] = new_weight

    # Add terms from relevant documents that are not in the original query
    for doc_id in relevant_docs:
        for term in relevant_docs[doc_id]:
            if term not in query_terms:
                p_r = sum(relevant_docs[doc].get(term, 0) for doc in relevant_docs) / len(
                    relevant_docs) if relevant_docs else 0
                p_nr = sum(non_relevant_docs[doc].get(term, 0) for doc in non_relevant_docs) / len(
                    non_relevant_docs) if non_relevant_docs else 0

                # Ensure probabilities are within valid range
                p_r = max(min(p_r, 1 - epsilon), epsilon)
                p_nr = max(min(p_nr, 1 - epsilon), epsilon)

                new_weight = math.log((p_r * 4 * (1 - p_nr)) / ((1 - p_r) * p_nr))
                new_query_terms[term] += beta * new_weight
                term_weights[term] = new_weight

    # Apply L1 regularization
    for term in new_query_terms:
        new_query_terms[term] -= l1_lambda * abs(term_weights.get(term, 0))

    return dict(new_query_terms)

def calculate_bm25_weights(coll, query_terms, df):
    """
    Calculate BM25 weights for terms in documents.

    Parameters:
    coll (DataColl): Collection of documents.
    query_terms (dict): Query terms and their frequencies.
    df (dict): Document frequencies for terms.

    Returns:
    dict: Dictionary of BM25 weights for documents and terms.
    """
    bm25_weights = {}
    k1 = 0.6
    b = 0.60
    N = coll.getNumOfDocs()
    avg_docLen = coll.getAvgDocLen()

    for doc_id, doc in coll.get_coll().items():
        termFreq = doc.get_term_list()
        bm25_weights[doc_id] = {}
        dl_avdl = doc.getDocLen() / avg_docLen
        K = k1 * ((1 - b) + (b * dl_avdl))

        for word_i, qfi in query_terms.items():
            ni = df.get(word_i, 0)
            fi = termFreq.get(word_i, 0)
            R = 0
            ri = 0
            k1 = 0.6
            k2 = 800
            num1 = (ri + 0.5) / (R - ri + 0.5)
            num2 = (ni - ri + 0.5) / (N * 2 - ni - R + ri + 0.5)
            k1_attr = ((k1 + 1) * fi) / (K + fi)
            k2_attr = ((k2 + 1) * qfi) / (k2 + qfi)
            bm25_weights[doc_id][word_i] = (math.log10(num1 / num2) * k1_attr * k2_attr)
    return bm25_weights

def build_relevance_model(top_k_docs, collection, collection_term_freq, total_doc_length, top_terms_count, percentage=0.9):
    """
    Build a relevance model from top k documents.

    Parameters:
    top_k_docs (list): List of top k document IDs.
    collection (dict): Collection of documents.
    collection_term_freq (dict): Term frequencies in the collection.
    total_doc_length (int): Total length of the collection.
    top_terms_count (int): Number of top terms to consider.
    percentage (float): Cumulative frequency percentage to include.

    Returns:
    dict: Relevance model with term probabilities.
    """
    term_counter = Counter()
    for doc_id in top_k_docs:
        terms = collection[doc_id].get_term_list()
        term_counter.update(terms)

    term_freq_pairs = term_counter.most_common(top_terms_count)
    total_freq = sum([freq for term, freq in term_freq_pairs])
    cumulative_sum = 0
    selected_terms = {}

    for term, freq in term_freq_pairs:
        cumulative_sum += freq
        selected_terms[term] = freq
        if cumulative_sum / total_freq >= percentage:
            break

    relevance_model = {term: count / sum(selected_terms.values()) for term, count in selected_terms.items()}
    return relevance_model

def my_prm(coll, query_id, query_terms, df, top_terms_count=245):
    """
    Perform Probabilistic Relevance Model (PRM) ranking.

    Parameters:
    coll (DataColl): Collection of documents.
    query_id (str): Query ID.
    query_terms (dict): Query terms and their frequencies.
    df (dict): Document frequencies for terms.
    top_terms_count (int): Number of top terms to consider.

    Returns:
    list: List of ranked documents and their scores.
    """
    lambda_param = 0.9
    k1 = 61  # Number of initial top documents considered
    k2 = 4  # Number of final top documents considered
    bm25_weights = calculate_bm25_weights(coll, query_terms, df)

    # Initial BM25 scoring
    bm25_weights = calculate_bm25_weights(coll, query_terms, df)
    initial_scores = {}
    for doc_id, term_weights in bm25_weights.items():
        score = sum(term_weights.values())
        initial_scores[doc_id] = score

    # Select top k1 documents
    sorted_initial_docs = sorted(initial_scores.items(), key=lambda item: item[1], reverse=True)
    top_k_docs_1 = [doc_id for doc_id, score in sorted_initial_docs[:k1]]

    # Prepare relevant and non-relevant documents for PRF
    relevant_docs = {doc_id: coll.get_coll()[doc_id].get_term_list() for doc_id in top_k_docs_1}
    non_relevant_docs = {doc_id: coll.get_coll()[doc_id].get_term_list() for doc_id in initial_scores if doc_id not in top_k_docs_1}

    # Apply Probabilistic Relevance Feedback
    updated_query_terms = probabilistic_relevance_feedback(query_terms, relevant_docs, non_relevant_docs, df, coll.getNumOfDocs())

    # Enhance query terms with pseudo-relevance feedback
    expanded_query_terms = pseudo_relevance_feedback(updated_query_terms, top_k_docs_1, coll, df)

    # Build initial relevance model
    relevance_model = build_relevance_model(top_k_docs_1, coll.get_coll(), df, coll.getTotalDocLen(), top_terms_count, 0.88)

    # Rerank using initial relevance model with BM25 weighting
    reranked_scores = {}
    for doc_id, doc in coll.get_coll().items():
        doc_terms = doc.get_term_list()
        doc_len = doc.getDocLen()
        collection_prob = {term: df.get(term, 0) / coll.getTotalDocLen() for term in relevance_model}
        p_w_d = {
            term: (1 - lambda_param) * (doc_terms.get(term, 0) / doc_len) + lambda_param * collection_prob.get(term, 1e-11)
            for term in relevance_model
        }
        bm25_weight = sum(bm25_weights[doc_id].get(term, 0) * 2 * expanded_query_terms.get(term, 0) for term in relevance_model)
        reranked_scores[doc_id] = sum([p_w_d[term] for term in relevance_model]) * bm25_weight

    # Select top k2 documents based on reranked scores
    sorted_reranked_docs = sorted(reranked_scores.items(), key=lambda item: item[1], reverse=True)
    top_k_docs_2 = [doc_id for doc_id, score in sorted_reranked_docs[:k2]]

    # Build final relevance model
    final_relevance_model = build_relevance_model(top_k_docs_2, coll.get_coll(), df, coll.getTotalDocLen(), top_terms_count, 0.98)

    # Final reranking using the final relevance model with BM25 weighting
    final_reranked_scores = {}
    for doc_id, doc in coll.get_coll().items():
        doc_terms = doc.get_term_list()
        doc_len = doc.getDocLen()
        p_w_d = {
            term: (1 - lambda_param) * (doc_terms.get(term, 0) / doc_len) + lambda_param * (df.get(term, 0) / coll.getTotalDocLen())
            for term in final_relevance_model
        }
        bm25_weight = sum(bm25_weights[doc_id].get(term, 0) * expanded_query_terms.get(term, 0) for term in final_relevance_model)
        final_reranked_scores[doc_id] = sum([p_w_d[term] for term in final_relevance_model]) * bm25_weight

    # Convert the final reranked scores to a list of tuples
    ranked_docs = sorted(final_reranked_scores.items(), key=lambda item: item[1], reverse=True)

    return ranked_docs

def bm25_ranking_my_prm(coll, query_terms, df):
    """
    Calculate BM25 ranking scores for a query (used in my_prm).

    Parameters:
    coll (DataColl): Collection of documents.
    query_terms (dict): Query terms and their frequencies.
    df (dict): Document frequencies for terms.

    Returns:
    dict: Dictionary of BM25 scores for documents.
    """
    bm25_weights = calculate_bm25_weights(coll, query_terms, df)
    scores = {}

    for doc_id, term_weights in bm25_weights.items():
        score = sum(term_weights.values())
        scores[doc_id] = score

    return scores

def pseudo_relevance_feedback(query_terms, top_docs, coll, df, num_feedback_docs=50, num_expansion_terms=200):
    """
    Perform pseudo-relevance feedback.

    Parameters:
    query_terms (dict): Query terms and their frequencies.
    top_docs (list): List of top document IDs.
    coll (DataColl): Collection of documents.
    df (dict): Document frequencies for terms.
    num_feedback_docs (int): Number of feedback documents to consider.
    num_expansion_terms (int): Number of expansion terms to add.

    Returns:
    dict: Expanded query terms and their weights.
    """
    term_scores = Counter()
    doc_count = Counter()

    # Calculate term frequencies in the top documents
    for doc_id in top_docs[:num_feedback_docs]:
        doc_terms = coll.get_coll()[doc_id].get_term_list()
        for term in doc_terms:
            term_scores[term] += doc_terms[term]
            doc_count[term] += 1

    # Calculate term importance using term frequency and document frequency
    term_importance = {}
    for term, score in term_scores.items():
        idf = math.log((coll.getNumOfDocs() + 1) / (doc_count[term] + 1))
        term_importance[term] = score * idf

    # Select the top terms based on term importance
    expansion_terms = sorted(term_importance.items(), key=lambda x: x[1], reverse=True)[:num_expansion_terms]

    # Expand the query with selected terms
    expanded_query_terms = query_terms.copy()
    for term, score in expansion_terms:
        if term not in expanded_query_terms:
            expanded_query_terms[term] = score

    return expanded_query_terms

def save_ranked_results(system_name, query_id, ranked_docs):
    """
    Save ranked results to a file.

    Parameters:
    system_name (str): Name of the ranking system.
    query_id (str): Query ID.
    ranked_docs (list): List of ranked documents and their scores.
    """
    output_filename = f"{system_name}_{query_id}Ranking.dat"
    output_filepath = os.path.join(outputFolder, output_filename)
    with open(output_filepath, 'w') as f:
        rank = 1
        for doc_id, score in ranked_docs:
            f.write(f"{query_id} Q0 {doc_id} {rank} {score} {system_name}\n")
            rank += 1
    print(f"Results for {query_id} saved to {output_filepath}")

def calculate_ap(ranked_docs, rel_docs):
    """
    Calculate Average Precision (AP).

    Parameters:
    ranked_docs (list): List of ranked documents and their scores.
    rel_docs (dict): Dictionary of relevant documents and their relevance scores.

    Returns:
    float: Average Precision (AP) score.
    """
    relevant_counts = 0
    precision_sum = 0
    for rank, (doc_id, _) in enumerate(ranked_docs, 1):
        if doc_id in rel_docs and rel_docs[doc_id] == 1:
            relevant_counts += 1
            precision_sum += relevant_counts / rank
    if relevant_counts == 0:
        return 0
    return precision_sum / relevant_counts

def calculate_p10(ranked_docs, rel_docs):
    """
    Calculate Precision at 10 (P@10).

    Parameters:
    ranked_docs (list): List of ranked documents and their scores.
    rel_docs (dict): Dictionary of relevant documents and their relevance scores.

    Returns:
    float: Precision at 10 (P@10) score.
    """
    relevant_counts = 0
    for rank, (doc_id, _) in enumerate(ranked_docs[:10], 1):
        if doc_id in rel_docs and rel_docs[doc_id] == 1:
            relevant_counts += 1
    return relevant_counts / 10

def calculate_dcg10(ranked_docs, rel_docs):
    """
    Calculate Discounted Cumulative Gain at 10 (DCG@10).

    Parameters:
    ranked_docs (list): List of ranked documents and their scores.
    rel_docs (dict): Dictionary of relevant documents and their relevance scores.

    Returns:
    float: DCG at 10 (DCG@10) score.
    """
    dcg = 0
    p = 10
    for i, (doc_id, _) in enumerate(ranked_docs[:p], 1):
        rel_score = rel_docs.get(doc_id, 0)
        if i == 1:
            dcg += rel_score
        else:
            dcg += rel_score / math.log2(i + 1)
    return dcg

def evaluate_all_queries(queries):
    """
    Evaluate all queries.

    Parameters:
    queries (dict): Dictionary of queries.

    Returns:
    dict: Dictionary of evaluation results (AP, P@10, DCG@10) for each model.
    """
    results = {'AP': {'BM25': {}, 'JM_LM': {}, 'My_PRM': {}},
               'P10': {'BM25': {}, 'JM_LM': {}, 'My_PRM': {}},
               'DCG10': {'BM25': {}, 'JM_LM': {}, 'My_PRM': {}}}
    for query_id in queries.keys():
        # BM25 evaluation
        ranked_file_bm25 = f"{outputFolder}/BM25_{query_id}Ranking.dat"
        if os.path.exists(ranked_file_bm25):
            ranked_docs_bm25 = []
            with open(ranked_file_bm25, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    doc_id = parts[2]
                    score = float(parts[4])
                    ranked_docs_bm25.append((doc_id, score))

            rel_docs, _ = get_re_docs(query_id)

            ap_bm25 = calculate_ap(ranked_docs_bm25, rel_docs)
            p10_bm25 = calculate_p10(ranked_docs_bm25, rel_docs)
            dcg10_bm25 = calculate_dcg10(ranked_docs_bm25, rel_docs)

            results['AP']['BM25'][query_id] = ap_bm25
            results['P10']['BM25'][query_id] = p10_bm25
            results['DCG10']['BM25'][query_id] = dcg10_bm25

        # JM_LM evaluation
        ranked_file_jm_lm = f"{outputFolder}/JM_LM_{query_id}Ranking.dat"
        if os.path.exists(ranked_file_jm_lm):
            ranked_docs_jm_lm = []
            with open(ranked_file_jm_lm, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    doc_id = parts[2]
                    score = float(parts[4])
                    ranked_docs_jm_lm.append((doc_id, score))

            rel_docs, _ = get_re_docs(query_id)

            ap_jm_lm = calculate_ap(ranked_docs_jm_lm, rel_docs)
            p10_jm_lm = calculate_p10(ranked_docs_jm_lm, rel_docs)
            dcg10_jm_lm = calculate_dcg10(ranked_docs_jm_lm, rel_docs)

            results['AP']['JM_LM'][query_id] = ap_jm_lm
            results['P10']['JM_LM'][query_id] = p10_jm_lm
            results['DCG10']['JM_LM'][query_id] = dcg10_jm_lm

        # PRM evaluation
        ranked_file_prm = f"{outputFolder}/My_PRM_{query_id}Ranking.dat"
        if os.path.exists(ranked_file_prm):
            ranked_docs_prm = []
            with open(ranked_file_prm, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    doc_id = parts[2]
                    score = float(parts[4])
                    ranked_docs_prm.append((doc_id, score))

            rel_docs, _ = get_re_docs(query_id)

            ap_prm = calculate_ap(ranked_docs_prm, rel_docs)
            p10_prm = calculate_p10(ranked_docs_prm, rel_docs)
            dcg10_prm = calculate_dcg10(ranked_docs_prm, rel_docs)

            results['AP']['My_PRM'][query_id] = ap_prm
            results['P10']['My_PRM'][query_id] = p10_prm
            results['DCG10']['My_PRM'][query_id] = dcg10_prm

    return results
def print_summary_statistics(results):
    """
    Print summary statistics for evaluation results.

    Parameters:
    results (dict): Dictionary of evaluation results (AP, P@10, DCG@10) for each model.
    """
    # Initialize dictionaries to store the data for each measure
    ap_data = {'Topic': []}
    p10_data = {'Topic': []}
    dcg10_data = {'Topic': []}

    # Initialize lists for models
    models = ['BM25', 'JM_LM', 'My_PRM']
    for model in models:
        ap_data[model] = []
        p10_data[model] = []
        dcg10_data[model] = []

    # Ensure topics are in order from R101 to R150
    topics = [f"R{str(i).zfill(3)}" for i in range(101, 151)]

    ap_data['Topic'] = topics
    p10_data['Topic'] = topics
    dcg10_data['Topic'] = topics

    # Populate the dictionaries with the results
    for topic in topics:
        for model in models:
            ap_data[model].append(results['AP'][model].get(topic, np.nan))
            p10_data[model].append(results['P10'][model].get(topic, np.nan))
            dcg10_data[model].append(results['DCG10'][model].get(topic, np.nan))

    # Convert the dictionaries to DataFrames
    ap_df = pd.DataFrame(ap_data)
    p10_df = pd.DataFrame(p10_data)
    dcg10_df = pd.DataFrame(dcg10_data)

    # Calculate the MAP, Average P@10, and Average DCG@10
    ap_df.loc['MAP'] = ap_df.mean(numeric_only=True)
    p10_df.loc['Average'] = p10_df.mean(numeric_only=True)
    dcg10_df.loc['Average'] = dcg10_df.mean(numeric_only=True)

    # Save the DataFrames to CSV files
    ap_df.to_csv('evaluation_results_ap.csv', index=False)
    p10_df.to_csv('evaluation_results_p10.csv', index=False)
    dcg10_df.to_csv('evaluation_results_dcg10.csv', index=False)

    # Print the DataFrames to the console (optional)
    print("Table 1. The performance of 3 models on average precision (MAP)")
    print(ap_df)

    print("\nTable 2. The performance of 3 models on precision@10")
    print(p10_df)

    print("\nTable 3. The performance of 3 models on DCG10")
    print(dcg10_df)

if __name__ == "__main__":
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
        print(f"Created directory: {outputFolder}")

    # only used by my_prm
    queries_long = load_queries_long(queryfile)
    print("Loaded Queries:", queries_long.keys())

    # only used by BM25 and JM smoothing
    queries = load_queries(queryfile)
    print("Loaded Queries:", queries.keys())

    folders = [folder for folder in os.listdir(data_collection_folder)]
    print(f"Processing folders: {folders}")

    for folder in folders:
        coll_folderpath = os.path.join(data_collection_folder, folder)
        if os.path.isfile(coll_folderpath) or not folder.startswith("Data_C"):
            continue  # Skip files and non-collection folders
        coll_num = folder[-3:]
        query_id = f"R{coll_num}"  # Ensure correct query ID format
        print(f"Calculating BM25-IR ranking scores for {folder} data collection")
        collections = parse_collection(coll_folderpath)
        if collections.getNumOfDocs() == 0:
            print(f"No documents found in collection {coll_folderpath}. Skipping.")
            continue
        avg_doc_len = avg_length(collections)
        df = my_df(collections.get_coll())
        if query_id in queries_long:  # Check for correct query ID
            rel_docs, R = get_re_docs(query_id)

            # Run BM25 and save results
            bm25_scores = bm25(collections, query_id, queries[query_id], df)
            ranked_docs = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
            save_ranked_results("BM25", query_id, ranked_docs)

            # Run JM_LM and save results
            jm_lm_scores = rank_documents_jm(query_id, queries[query_id], collections.get_coll(), df, collections.getTotalDocLen())
            save_ranked_results("JM_LM", query_id, jm_lm_scores)

            # Run BM25 ranking and save results for MY_PRM
            bm25_scores = bm25_ranking_my_prm(collections, queries_long[query_id], df)
            ranked_docs = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
            # Run PRF and then BM25 for MY_PRM
            top_docs = [doc_id for doc_id, _ in list(bm25_scores.items())[:5]]
            expanded_query_terms = pseudo_relevance_feedback(queries_long[query_id], top_docs, collections, df)
            bm25_scores_prf = bm25_ranking_my_prm(collections, expanded_query_terms, df)
            ranked_docs_prf = sorted(bm25_scores_prf.items(), key=lambda x: x[1], reverse=True)
            # Call my_prm with long query terms and PRF
            prm_scores = my_prm(collections, query_id, queries_long[query_id], df)
            ranked_docs_prm = sorted(prm_scores, key=lambda x: x[1], reverse=True)
            save_ranked_results("My_PRM", query_id, ranked_docs_prm)
        else:
            print(f"Query for collection {coll_num} not found in queries_long.")

    results = evaluate_all_queries(queries_long)
    print_summary_statistics(results)
