"""
Task 1 – Design a BM25-based IR model (BM25)
that ranks documents in each data collection using the corresponding topic (query) for all 50 data collections.
"""

import os
import math
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
        Initialize a DataDoc instance.

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
        Add a term to the term list.

        Parameters:
        term (str): Term to be added.
        """
        if term not in self.terms:
            self.terms[term] = 1
        else:
            self.terms[term] += 1

class DataColl:
    def __init__(self):
        """
        Initialize a DataColl instance.
        """
        self.collections = {}
        self.numOfDocs = 0
        self.totalDocLength = 0
        self.avgDocLen = 0

    def add_doc(self, doc):
        """
        Add a document to the collection.

        Parameters:
        doc (DataDoc): Document to be added.
        """
        self.collections[doc.getDocId()] = doc
        self.totalDocLength += doc.getDocLen()
        self.numOfDocs += 1

    def get_coll(self):
        """
        Get the collection.

        Returns:
        dict: Dictionary of documents in the collection.
        """
        return self.collections

    def getTotalDocLen(self):
        """
        Get the total document length of the collection.

        Returns:
        int: Total document length.
        """
        return self.totalDocLength

    def getNumOfDocs(self):
        """
        Get the number of documents in the collection.

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

# Function to parse a query
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
        term = stemmer.stem(original_term)  # Ensure stemming
        if len(term) > 2 and term not in stop_words:
            curr_word[term] = curr_word.get(term, 0) + 1
        else:
            print(f"Excluded term: {original_term} (stemmed: {term})")  # Debugging excluded terms
    print(f"Parsed query terms: {curr_word}")  # Debug statement for parsed query terms
    return curr_word

# Function to load queries from a file and parse them
def load_queries(query_file):
    """
    Load and parse queries from a file.

    Parameters:
    query_file (str): Path to the query file.

    Returns:
    dict: Dictionary of queries with query IDs as keys and parsed query terms as values.
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
                r'<narr> Narrative:\s*(.*?)\s*'  # Commented out line
                r'</Query>', re.DOTALL
            )
            matches = query_pattern.findall(query_data)
            for query_id, title, description, narrative in matches:
                title_query = f"{title}"
                parsed_query = parse_query(title_query)
                queries[query_id.strip()] = parsed_query
                print(f"Loaded and parsed query {query_id}: {parsed_query}")  # Debug statement for loaded query
    except Exception as e:
        print(f"Failed to load queries from {query_file}: {e}")
    return queries

# Parsing documents in a collection
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
                        stemmed_term = stemmer.stem(term.lower())  # Ensure stemming
                        word_count += 1
                        if len(stemmed_term) > 2 and stemmed_term not in stop_words:
                            dataDoc.add_term(stemmed_term)
        dataDoc.set_doc_len(word_count)
        coll.add_doc(dataDoc)
    return coll

# Calculate document-frequency for given data collection
def my_df(coll):
    """
    Calculate document frequency for a given data collection.

    Parameters:
    coll (dict): Dictionary of documents in the collection.

    Returns:
    dict: Dictionary of terms and their document frequencies.
    """
    docFreq = {}
    for _, doc in coll.items():
        terms = doc.get_term_list()
        for term in terms:
            docFreq[term] = docFreq.get(term, 0) + 1
    return docFreq

# Calculate and return avg length of all docs in a data coll
def avg_length(coll):
    """
    Calculate and return the average length of all documents in a data collection.

    Parameters:
    coll (DataColl): DataColl object containing the collection.

    Returns:
    float: Average document length.
    """
    avg_length = coll.getTotalDocLen() / coll.getNumOfDocs()
    coll.setAvgDocLen(avg_length)
    return avg_length

def bm25(coll, query_id, query_terms, df):
    """
    Calculate BM25 ranking scores for a collection of documents.

    Parameters:
    coll (DataColl): DataColl object containing the collection.
    query_id (str): Query ID.
    query_terms (dict): Dictionary of query terms and their frequencies.
    df (dict): Dictionary of document frequencies for terms.

    Returns:
    dict: Dictionary of document IDs and their BM25 scores.
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
        #    ri = relevant_ni.get(word_i, 0)  # Number of relevant documents containing the term
            num1 = (ri + 0.5) / (R - ri + 0.5)
            num2 = (ni - ri + 0.5) / (N * 2 - ni - R + ri + 0.5)
            k1_attr = ((k1 + 1) * fi) / (K + fi)
            k2_attr = ((k2 + 1) * qfi) / (k2 + qfi)
            scores[docId] += (math.log10(num1 / num2) * k1_attr * k2_attr)

    return scores

# Save ranked results to file
def save_ranked_results(system_name, query_id, ranked_docs):
    """
    Save ranked results to a file.

    Parameters:
    system_name (str): Name of the ranking system.
    query_id (str): Query ID.
    ranked_docs (list): List of tuples containing document IDs and their scores.

    Returns:
    None
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
    # Ensure the output directory exists
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
        print(f"Created directory: {outputFolder}")

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
            bm25_scores = bm25(collections, query_id, queries[query_id], df)
            ranked_docs = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
            print(f"Top 10 ranked documents for query {query_id}: {ranked_docs[:10]}")
            save_ranked_results("BM25", query_id, ranked_docs)
        else:
            print(f"Query for collection {coll_num} not found in queries.")
    print("Completed!! The ranking scores are saved in the RankingOutputs folder.")
