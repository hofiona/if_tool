"""
Task 2 – Design a Jelinek-Mercer based Language Model (JM_LM)
that ranks documents in each data collection using the corresponding topic (query) for all 50 data collections.

"""


import os
import re
import string
from nltk.stem import PorterStemmer

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


if __name__ == "__main__":
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
        print(f"Created directory: {outputFolder}")

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
        if query_id in queries:  # Check for correct query ID

            # Run JM_LM and save results
            jm_lm_scores = rank_documents_jm(query_id, queries[query_id], collections.get_coll(), df, collections.getTotalDocLen())
            save_ranked_results("JM_LM", query_id, jm_lm_scores)

        else:
            print(f"Query for collection {coll_num} not found in queries_long.")
