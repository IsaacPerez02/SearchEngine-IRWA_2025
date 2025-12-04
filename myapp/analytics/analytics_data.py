import json
import random
import altair as alt
import pandas as pd
import os

class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # Example of statistics table
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    fact_clicks = dict() 
    dwell_start_times = dict() 
    query_results = dict() 

    ### Please add your custom tables here:

    def save_query_terms(self, terms: str) -> int:
        search_id = random.randint(0, 100000)
        search_information = {
            "search_id": search_id,
            "query": terms,
            "n_terms": len(terms.split()),
            "timestamp": str(pd.Timestamp.now())
        }
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", "queries.jsonl")
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(search_information, ensure_ascii=False) + "\n")
        return search_id
    
    def save_document_click(self, doc_id, search_id, rank, session_id, dwell_time=None):
        """
        Save a clicked document event.
        """
        click_data = {
            "doc_id": doc_id,
            "search_id": search_id,
            "rank": rank,
            "session_id": session_id,
            "timestamp": str(pd.Timestamp.now()),
            "dwell_time": dwell_time
        }
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", "doc_clicks.jsonl")
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(click_data, ensure_ascii=False) + "\n")
        # update in-memory counter
        self.fact_clicks[doc_id] = self.fact_clicks.get(doc_id, 0) + 1

    def start_dwell_timer(self, session_id, doc_id):
        self.dwell_start_times[f"{session_id}_{doc_id}"] = pd.Timestamp.now().timestamp()

    def end_dwell_timer(self, session_id, doc_id):
        key = f"{session_id}_{doc_id}"
        start_ts = self.dwell_start_times.get(key)
        if start_ts:
            dwell_time = pd.Timestamp.now().timestamp() - start_ts
            del self.dwell_start_times[key]
            return dwell_time
        return None
    
    def plot_number_of_views(self):
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)

        chart = alt.Chart(df).mark_bar().encode(
            y=alt.Y('Document ID', sort='-x'),
            x='Number of Views'
        ).properties(
            title='Number of Views per Document (Horizontal)'
        )
        return chart.to_html()


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
