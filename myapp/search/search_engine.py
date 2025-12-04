import random
import numpy as np
import os
import json

from myapp.search.objects import Document
from myapp.core.utils import build_terms
from myapp.search.algorithms import create_index_tfidf_corpus
from myapp.search.algorithms import rank_products

def dummy_search(corpus: dict, search_id, num_results=20):
    """
    Just a demo method, that returns random <num_results> documents from the corpus
    :param corpus: the documents corpus
    :param search_id: the search id
    :param num_results: number of documents to return
    :return: a list of random documents from the corpus
    """
    res = []
    doc_ids = list(corpus.keys())
    docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
    for doc_id in docs_to_return:
        doc = corpus[doc_id]
        res.append(Document(pid=doc.pid, title=doc.title, description=doc.description,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=random.random()))
    return res

class SearchEngine:

    def __init__(self):
        self.index = None
        self.tf = None
        self.df = None
        self.idf = None
        self.title_index = None

    def build_index(self, corpus):
        print("Building index...")
        (self.index, self.tf, self.df, self.idf, self.title_index) = create_index_tfidf_corpus(corpus)

    def search_in_corpus(self, query, corpus, search_id=None):
        query = build_terms(query)
        docs = None

        for term in query:
            if term in self.index:
                term_docs = {posting[0] for posting in self.index[term]}
                docs = term_docs if docs is None else docs & term_docs
            else:
                docs = set()
                break

        docs = list(docs)
        ranked_products = rank_products(query, docs, self.index, self.idf, self.tf, self.title_index)
        top = 10

        results = []
        rank = 0
        for pid in ranked_products[:top]:
            doc = corpus.get(pid)
            if doc:
                result = Document(
                    pid=doc.pid,
                    title=doc.title,
                    description=doc.description,
                    url=f"doc_details?pid={doc.pid}&search_id={search_id or 'N/A'}&param2=2",
                    ranking=rank
                )
                rank += 1
                results.append(result)
        return results

    def search(self, search_query, corpus, search_id=None):
        print("Search query:", search_query)
        print()
        return self.search_in_corpus(search_query, corpus, search_id)
