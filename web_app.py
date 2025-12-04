import os
import random
import json
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
import nltk

load_dotenv()  # take environment variables from .env
nltk.download('stopwords')


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)
# Log first element of corpus to verify it loaded correctly:
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])

if search_engine.index is None:
    search_engine.build_index(corpus)

# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']
    session['last_search_query'] = search_query

    # Generar un search_id
    search_id = analytics_data.save_query_terms(search_query)
    session['last_search_id'] = search_id  # para doc_details

    # Ejecutar búsqueda
    results = search_engine.search(search_query, corpus, search_id)

    # Guardar ranking de resultados en memoria
    ranked_doc_ids = [doc.pid for doc in results]
    analytics_data.query_results[search_id] = ranked_doc_ids

    # Guardar búsqueda en JSONL
    os.makedirs("data", exist_ok=True)
    search_data = {
        "search_id": search_id,
        "query": search_query,
        "n_terms": len(search_query.split()),
        "found_count": len(results),
        "result_doc_ids": ranked_doc_ids
    }
    with open(os.path.join("data", "searches.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(search_data, ensure_ascii=False) + "\n")

    return render_template('results.html',
                           results_list=results,
                           page_title="Results",
                           found_counter=len(results),
                           rag_response=rag_generator.generate_response(search_query, results))

@app.route('/doc_details', methods=['GET'])
def doc_details():
    clicked_doc_id = request.args["pid"]
    last_search_id = session.get("last_search_id")

    # Actualizar clicks en memoria
    if clicked_doc_id in analytics_data.fact_clicks:
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1
    click_count = analytics_data.fact_clicks[clicked_doc_id]

    # Obtener ranking
    rank = None
    if last_search_id and analytics_data.query_results.get(last_search_id):
        try:
            rank = analytics_data.query_results[last_search_id].index(clicked_doc_id) + 1
        except ValueError:
            rank = None

    os.makedirs("data", exist_ok=True)
    dwell_time = analytics_data.end_dwell_timer(session.sid if hasattr(session, 'sid') else str(random.randint(0,100000)), clicked_doc_id)
    click_data = {
        "doc_id": clicked_doc_id,
        "search_id": last_search_id,
        "rank": rank,
        "click_count": click_count,
        "dwell_time": dwell_time,
    }
    with open(os.path.join("data", "doc_clicks.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(click_data, ensure_ascii=False) + "\n")

    # Iniciar timer de dwell para próxima visita
    analytics_data.start_dwell_timer(session.sid if hasattr(session, 'sid') else str(random.randint(0,100000)), clicked_doc_id)

    # Renderizar documento
    item = corpus.get(clicked_doc_id)
    return render_template('doc_details.html', doc=item)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[doc_id]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    for doc in visited_docs: print(doc)
    return render_template('dashboard.html', visited_docs=visited_docs)


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
