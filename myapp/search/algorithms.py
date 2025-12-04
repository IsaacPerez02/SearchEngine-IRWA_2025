import collections
from collections import defaultdict
from array import array
from typing import Dict
import numpy as np
from numpy import linalg as la
import math
from myapp.search.objects import Document
from myapp.core.utils import build_terms

def create_index_tfidf_corpus(corpus: Dict[str, Document]):
    print("Building tfidf index...")
    index = defaultdict(list)       
    tf = defaultdict(list)         
    df = defaultdict(int)           
    idf = defaultdict(float)        
    title_index = {}
    num_products = len(corpus)

    # Loop through each Document object
    for pid, doc in corpus.items():

        # --- Preprocess fields ---
        title_tokens = build_terms(doc.title or "")
        description_tokens = build_terms(doc.description or "")
        
        # Combine for indexing
        words = title_tokens + description_tokens

        title_index[pid] = doc.title or ""

        # Build per-document postings: term -> [pid, positions]
        current_product_index = {}

        for position, term in enumerate(words):
            if term not in current_product_index:
                current_product_index[term] = [pid, array('I')]
            current_product_index[term][1].append(position)

        # Compute TF normalization factor
        norm = math.sqrt(sum(len(posting[1]) ** 2 for posting in current_product_index.values()))

        # Update TF and DF
        for term, posting in current_product_index.items():
            tf[term].append(np.round(len(posting[1]) / norm, 4))
            df[term] += 1

        # Add postings to global index
        for term, posting in current_product_index.items():
            index[term].append(posting)

    # Compute IDF
    for term in df:
        idf[term] = np.round(np.log(num_products / df[term]), 4)

    return index, tf, df, idf, title_index

def rank_products(terms, docs, index, idf, tf, title_index):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    title_index -- mapping between page id and page title

    Returns:
    Print the list of ranked documents
    """

    
    # I'm interested only on the element of the docVector corresponding to the query terms
    # The remaining elements would became 0 when multiplied to the query_vector
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # I call doc_vectors[k] for a nonexistent key k, the key-value pair (k,[0]*len(terms)) will be automatically added to the dictionary
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms)  # get the frequency of each term in the query.
    # Example: collections.Counter(["hello","hello","world"]) --> Counter({'hello': 2, 'world': 1})
    # HINT: use when computing tf for query_vector

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        ## Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex]= query_terms_count[term]/query_norm * idf[term] #query_vector[0] corresponds to the first term in the query

        # Generate doc_vectors for matching docs
        for doc_index, (doc, postings) in enumerate(index[term]):
            # Example of [doc_index, (doc, postings)]
            # 0 (26, array('I', [1, 4, 12, 15, 22, 28, 32, 43, 51, 68, 333, 337]))
            # 1 (33, array('I', [26, 33, 57, 71, 87, 104, 109]))
            # term is in doc 26 in positions 1,4, .....
            # term is in doc 33 in positions 26,33, .....

            #tf[term][0] will contain the tf of the term "term" in the doc 26
            if doc in docs: #if the odcument is in the list of documents retrieved (matching the query)
                doc_vectors[doc][termIndex] = tf[term][doc_index] * idf[term]  # TODO: check if multiply for idf

    # Calculate the score of each doc
    # compute the cosine similarity between queyVector and each docVector:
    # HINT: you can use the dot product because in case of normalized vectors it corresponds to the cosine similarity
    # see np.dot

    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    #print document titles instead if document id's
    #result_docs=[ title_index[x] for x in result_docs ]

    #print ('\n'.join(result_docs), '\n')
    return result_docs

def create_index_bm25_corpus(corpus: Dict[str, Document]):
    """
    Build inverted index and BM25 statistics for a corpus of Document objects.
    corpus: dict of pid -> Document
    Returns: index, tf, df, idf, title_index, doc_len, avg_doc_len
    """
    print("Building bm25 index...")
    index = defaultdict(list)       # term -> postings list [pid, positions]
    tf = defaultdict(list)          # term -> raw frequency counts across documents
    df = defaultdict(int)           # term -> number of documents containing term
    idf = {}                        # BM25 idf values
    title_index = {}                # pid -> title
    doc_len = {}                    # pid -> document length (# of terms)
    N = len(corpus)

    for pid, doc in corpus.items():
        title_tokens = build_terms(doc.title or "")
        description_tokens = build_terms(doc.description or "")
        words = title_tokens + description_tokens

        title_index[pid] = doc.title or ""
        doc_len[pid] = len(words)

        term_positions = {}
        for pos, term in enumerate(words):
            term_positions.setdefault(term, []).append(pos)

        for term, positions in term_positions.items():
            index[term].append([pid, array('I', positions)])
            df[term] += 1
            tf[term].append(len(positions))

    # BM25-style IDF
    for term, freq in df.items():
        idf[term] = math.log(N / freq)

    avg_doc_len = sum(doc_len.values()) / N if N > 0 else 0
    return index, tf, df, idf, title_index, doc_len, avg_doc_len


def rank_products_bm25(terms, docs, index, idf, tf, doc_len, avg_doc_len, k1=1.5, b=0.75):
    """
    Rank documents using BM25 scoring.
    terms: list of query terms
    docs: list of candidate doc IDs
    Returns: list of ranked doc IDs
    """
    scores = defaultdict(float)

    for term in terms:
        if term not in index:
            continue
        postings = index[term]
        tf_list = tf[term]
        idf_value = idf[term]

        for i, (doc_id, positions) in enumerate(postings):
            if doc_id not in docs:
                continue
            f = tf_list[i]          # raw frequency of term in this doc
            dl = doc_len[doc_id]    # length of this document
            denom = f + k1 * (1 - b + b * dl / avg_doc_len)
            score = idf_value * ((f * (k1 + 1)) / denom)
            scores[doc_id] += score

    doc_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_scores]
