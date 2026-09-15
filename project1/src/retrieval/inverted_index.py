from src.preprocessing.dataset import preprocess_cases
from src.retrieval.linked_lists import InvertedIndexNode
from src.retrieval.linked_lists import extract_doc_key
from pathlib import Path
from collections import Counter
from src.RankedSearch.rankedsearch import InverseDocumentFrequency

class InvertedIndex:
    def __init__(self, term: str):
        self.term = term # termo do vocabulário
        self.postings_head = None # cabeça da lista ligada de Node
        self.postings_tail = None # cauda da lista ligada de Node
        self.n = 0 # número de documentos que contém o termo (document frequency)
        self.idf = 0.0 # valor de IDF do termo
        self.is_stopword = False # flag marca se é stopword

    def add_posting(self, case_id: str, tf: int = 1) -> None:
        """
        Adiciona um case_id à lista ligada com sua frequência de termo (tf).
        Como os casos são iterados em ordem crescente de doc_key,
        basta anexar na cauda (tail).
        """
        self.n += 1
        new_node = InvertedIndexNode(case_id, tf)
        if self.postings_head is None:
            self.postings_head = new_node
            self.postings_tail = new_node
        else:
            self.postings_tail.next = new_node
            self.postings_tail = new_node

    def print_inverted_index(self) -> None:
        """Imprime o índice invertido para inspeção."""
        print(f"Termo: {self.term}")
        current = self.postings_head
        while current is not None:
            print(
                f"  Case ID: {current.data[0]}, "
                f"doc_key: {current.doc_key}, "
                f"TF: {current.data[1]}, "
                f"IDF: {self.idf:.4f}, "
                f"Stopword: {self.is_stopword}"
            )
            current = current.next

        print(f"Total de postings para o termo {self.term}: {self.n}")

    def get_postings(self) -> list[str]:
        """
        Retorna apenas a lista de identificadores (case_id) presentes 
        na lista ligada, preservando a ordem crescente.
        """
        result = []
        current = self.postings_head
        while current is not None:
            # current.data armazena a tupla (case_id, tf)
            result.append(current.data[0])
            current = current.next
        return result

    def get_postings_with_frequencies(self) -> list[tuple[str, int]]:
        """
        Retorna uma lista de tuplas (case_id, tf) com os identificadores 
        e suas respectivas frequências no documento.
        Utilizado no cálculo de pesos para TF-IDF / Ranked Retrieval.
        """
        result = []
        current = self.postings_head
        while current is not None:
            # current.data armazena a tupla (case_id, tf)
            result.append((current.data[0], current.data[1]))
            current = current.next
        return result

    def compute_idf(self, total_docs: int) -> None:
        """
        Calcula e armazena o IDF do termo dado o número total de documentos.
        """
        if self.n <= 0:
            self.idf = 0.0
            return

        self.idf = InverseDocumentFrequency(self.n, total_docs)

    def mark_as_stopword(self, value: bool = True) -> None:
        self.is_stopword = bool(value)

def build_index_from_cases(path: str | Path) -> dict[str, InvertedIndex]:
    """
    Carrega casos com preprocess_cases e constrói um índice invertido.
    Ordena os documentos por doc_key e calcula as frequências locais
    com Counter antes de adicionar às listas de postings.
    """
    cases = preprocess_cases(path=path)
    valid_cases = [c for c in cases if c.get("case_id") is not None]
    valid_cases.sort(key=lambda c: extract_doc_key(c["case_id"]))

    inverted_index = {}

    for case in valid_cases:
        case_id = case["case_id"]
        # calcula a contagem exata de cada token único dentro do caso
        token_counts = Counter(case.get("tokens", []))

        for token, tf in token_counts.items():
            if token not in inverted_index:
                inverted_index[token] = InvertedIndex(token)

            inverted_index[token].add_posting(case_id, tf=tf)

    return inverted_index

def finalize_index(
    inverted_index: dict[str, InvertedIndex],
    total_docs: int,
    idf_threshold: float | None = None,
    max_df_frac: float | None = None,
) -> None:
    """
    Calcula IDF para cada termo e marca stopwords.

    - `idf_threshold`: marca como stopword termos com `idf < idf_threshold`.
    - `max_df_frac`: marca como stopword termos com (`df / total_docs`) >= `max_df_frac`.

    Ao menos um dos thresholds deve ser fornecido; se ambos forem None,
    apenas calcula e armazena os idf sem marcar stopwords.
    """
    for term, index in inverted_index.items():
        index.compute_idf(total_docs)

    if idf_threshold is None and max_df_frac is None:
        return

    for term, index in inverted_index.items():
        df_frac = index.n / total_docs if total_docs > 0 else 0.0

        mark = False
        if idf_threshold is not None and index.idf < idf_threshold:
            mark = True
        if max_df_frac is not None and df_frac >= max_df_frac:
            mark = True

        index.mark_as_stopword(mark)

    return inverted_index

if __name__ == "__main__":
    PATH = Path("data/raw/cases.csv")
    cases = preprocess_cases(PATH)
    total_docs = len([c for c in cases if c.get("case_id")])
    n_stopwords = 0

    inverted_index = build_index_from_cases(PATH)
    # marca stopwords: termos com idf < 0.1 ou que aparecem em >=80% dos docs
    inverted_index = finalize_index(inverted_index, total_docs, idf_threshold=0.1, max_df_frac=0.8)
    for term, index in inverted_index.items():
        index.print_inverted_index()
        if index.is_stopword:
            n_stopwords += 1

    print(f"\nTotal de documentos processados: {total_docs}")
    print(f"Total de termos no índice invertido: {len(inverted_index)}")
    print(f"Total de stopwords no índice invertido: {n_stopwords}")