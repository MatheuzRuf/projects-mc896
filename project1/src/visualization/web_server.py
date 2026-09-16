import csv
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.visualization.graph import load_case_graph
from src.preprocessing.dataset import preprocess_cases
from src.retrieval.inverted_index import build_index_from_cases, finalize_index
from src.RankedSearch.rankedsearch import TF_IDF, RankedSearch
from src.retrieval.cases_linked_list import build_cases_list_from_cases
from src.retrieval.boolean_retrieval import AND as BOOL_AND, OR as BOOL_OR, NOT as BOOL_NOT

def list_case_ids(nodes_path: str | Path) -> list[str]:
    """Lista os case_ids na mesma ordem em que aparecem em nodes.csv."""

    case_ids = []
    seen = set()

    with Path(nodes_path).open("r", encoding="utf-8", newline="") as file:
        for node in csv.DictReader(file):
            case_id = node["node_id"].split(":", 1)[0]
            if case_id not in seen:
                seen.add(case_id)
                case_ids.append(case_id)

    return case_ids

def create_request_handler(
    nodes_path: str | Path,
    edges_path: str | Path,
    static_dir: str | Path,
    cases_path: str | Path,
):
    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)
    static_dir = Path(static_dir)

    class GraphRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, data, status=200):
            content = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_static(self, filename, content_type):
            path = static_dir / filename
            if not path.exists():
                self.send_error(404)
                return

            content = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            request = urlparse(self.path)

            if request.path == "/api/cases":
                self._send_json({"cases": list_case_ids(nodes_path)})
                return

            if request.path == "/api/graph":
                query = parse_qs(request.query)
                case_id = query.get("case_id", [None])[0]

                try:
                    selected_case, nodes, edges = load_case_graph(
                        nodes_path,
                        edges_path,
                        case_id,
                    )
                except ValueError as error:
                    self._send_json({"error": str(error)}, status=404)
                    return

                self._send_json(
                    {
                        "case_id": selected_case,
                        "nodes": nodes,
                        "edges": edges,
                    }
                )
                return

            # detalhes do caso (texto original, normalizado, tokens, stopwords filtradas)
            if request.path == "/api/case_details":
                query = parse_qs(request.query)
                case_id = query.get("case_id", [None])[0]

                cases = preprocess_cases(cases_path)
                selected = None
                for c in cases:
                    if c.get("case_id") == case_id:
                        selected = c
                        break

                if selected is None:
                    self._send_json({"error": "Caso não encontrado."}, status=404)
                    return

                # build inverted index to identify stopwords
                inverted_index = build_index_from_cases(cases_path)
                total_docs = len([c for c in cases if c.get("case_id")])
                inverted_index = finalize_index(inverted_index, total_docs, idf_threshold=0.1, max_df_frac=0.8)

                tokens = selected.get("tokens", [])
                stopwords_in_case = [t for t in tokens if t in inverted_index and getattr(inverted_index[t], "is_stopword", False)]
                retrieval_tokens = [t for t in tokens if not (t in inverted_index and getattr(inverted_index[t], "is_stopword", False))]

                self._send_json(
                    {
                        "case_id": case_id,
                        "original_text": selected.get("original_text") or selected.get("case_text") or "",
                        "normalized_text": selected.get("normalized_text", ""),
                        "tokens": tokens,
                        "stopwords": stopwords_in_case,
                        "retrieval_tokens": retrieval_tokens,
                    }
                )
                return

            # TF-IDF matrix (terms x documents)
            if request.path == "/api/tfidf":
                cases = preprocess_cases(cases_path)
                total_docs = len([c for c in cases if c.get("case_id")])
                inverted_index = build_index_from_cases(cases_path)
                inverted_index = finalize_index(inverted_index, total_docs, idf_threshold=0.1, max_df_frac=0.8)

                # terms order (skip stopwords)
                terms = [term for term, idx in inverted_index.items() if not getattr(idx, "is_stopword", False)]

                tfidf_result = TF_IDF(inverted_index, total_docs, query="")

                # document order
                documents = sorted([name for name in tfidf_result.keys() if name != "query"])

                # build matrix: rows = terms, cols = documents
                matrix = []
                for i in range(len(terms)):
                    row = [tfidf_result[doc][i] for doc in documents]
                    matrix.append(row)

                self._send_json({"terms": terms, "documents": documents, "matrix": matrix})
                return

            # retrieval endpoint: supports mode=ranked|boolean
            if request.path == "/api/retrieve":
                query = parse_qs(request.query)
                q = query.get("q", [""])[0]
                mode = query.get("mode", ["ranked"])[0]
                k = int(query.get("k", [10])[0])

                cases = preprocess_cases(cases_path)
                total_docs = len([c for c in cases if c.get("case_id")])
                inverted_index = build_index_from_cases(cases_path)
                inverted_index = finalize_index(inverted_index, total_docs, idf_threshold=0.1, max_df_frac=0.8)

                if mode == "ranked":
                    tfidf_result = TF_IDF(inverted_index, total_docs, query=q)
                    ranking = RankedSearch(tfidf_result)
                    # ranking is list of (score, docname)
                    resp = [{"case_id": doc, "score": float(score)} for score, doc in ranking[:k]]
                    self._send_json({"results": resp})
                    return

                if mode == "boolean":
                    # simple left-to-right boolean parser supporting AND, OR, NOT
                    tokens = q.split()
                    if not tokens:
                        self._send_json({"results": []})
                        return

                    cases_list = build_cases_list_from_cases(cases_path)

                    def postings_set(term: str) -> set:
                        idx = inverted_index.get(term)
                        return set(idx.get_postings()) if idx else set()

                    i = 0
                    current_set = None
                    try:
                        while i < len(tokens):
                            tok = tokens[i].upper()
                            if tok == "NOT":
                                term = tokens[i + 1]
                                s = BOOL_NOT(inverted_index, term, cases_list)
                                i += 2
                            elif tok in ("AND", "OR"):
                                # operator; apply with next term
                                op = tok
                                term = tokens[i + 1]
                                s_next = postings_set(term)
                                if op == "AND":
                                    current_set = (current_set & s_next) if current_set is not None else s_next
                                else:
                                    current_set = (current_set | s_next) if current_set is not None else s_next
                                i += 2
                                continue
                            else:
                                # plain term
                                term = tokens[i]
                                s = postings_set(term)
                                i += 1

                            if current_set is None:
                                current_set = s
                            else:
                                # if previous token was an operator it was already applied
                                current_set = current_set & s

                        resp = [{"case_id": c} for c in sorted(current_set)]
                        self._send_json({"results": resp})
                        return
                    except Exception:
                        self._send_json({"error": "Erro ao processar consulta booleana."}, status=400)
                        return

            static_files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            }
            static_file = static_files.get(request.path)
            if static_file:
                self._send_static(*static_file)
                return

            self.send_error(404)

        def log_message(self, format, *args):
            return

    return GraphRequestHandler

def serve_graph_interface(
    nodes_path: str | Path,
    edges_path: str | Path,
    static_dir: str | Path,
    cases_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Executa a interface local ate o usuario interromper o processo."""

    handler = create_request_handler(nodes_path, edges_path, static_dir, cases_path)
    server = ThreadingHTTPServer((host, port), handler)

    print(f"Graph interface running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping graph interface.")
    finally:
        server.server_close()

__all__ = [
    "list_case_ids",
    "create_request_handler",
    "serve_graph_interface",
]
