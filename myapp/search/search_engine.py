from myapp.core.utils import build_terms
from myapp.search.objects import Document
from myapp.search.algorithms import (
    rank_products,
    rank_products_bm25
)

class SearchEngine:
    def search(self, query: str, corpus: dict, index_data: dict, search_id, mode):
        """
        Stateless search function.
        - query: raw query string
        - corpus: dict of pid -> Document
        - index_data: dict containing all index variables + 'mode' flag
        - search_id: optional search identifier
        - mode: 'tfidf' or 'bm25'
        """

        query_terms = build_terms(query)

        # collect candidate docs (intersection of query terms)
        docs = None
        for term in query_terms:
            if term not in index_data["index"]:
                docs = set()
                break
            term_docs = {posting[0] for posting in index_data["index"][term]}
            docs = term_docs if docs is None else docs & term_docs

        docs = list(docs or [])

        # choose ranking strategy
        if mode == "tfidf":
            ranked_pids = rank_products(
                query_terms,
                docs,
                index_data["index"],
                index_data["idf"],
                index_data["tf"],
                index_data["title_index"]
            )
        elif mode == "bm25":
            ranked_pids = rank_products_bm25(
                query_terms,
                docs,
                index_data["index"],
                index_data["idf"],
                index_data["tf"],
                index_data["doc_len"],
                index_data["avg_doc_len"],
                index_data.get("k1", 1.5),
                index_data.get("b", 0.75)
            )
        else:
            raise ValueError(f"Unknown search mode: {mode}")

        # wrap results into Document objects
        results = []
        for rank, pid in enumerate(ranked_pids[:10]):
            doc = corpus.get(pid)
            if doc:
                results.append(Document(
                    pid=doc.pid,
                    title=doc.title,
                    description=doc.description,
                    url=f"doc_details?pid={doc.pid}&search_id={search_id or 'N/A'}",
                    average_rating=doc.average_rating,
                    selling_price=doc.selling_price,
                    out_of_stock=doc.out_of_stock,
                    ranking=rank
                ))
        return results
