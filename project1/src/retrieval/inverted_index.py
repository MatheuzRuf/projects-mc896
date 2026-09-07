from ..preprocessing.dataset import preprocess_cases
from .linked_list import Node
from pathlib import Path

class InvertedIndex:
    """Índice invertido simples onde postings é uma lista ligada de Node.

    Cada Node.case_id contém o identificador do caso conforme presente
    na planilha CSV (campo case_id).
    """

    def __init__(self, term: str):
        self.term = term
        self.postings_head = None

    def add_posting(self, case_id: str) -> None:
        """Adiciona um case_id à lista ligada se ainda não existir."""

        # percorre a lista para verificar duplicata
        current = self.postings_head
        while current is not None:
            if current.case_id == case_id:
                return
            current = current.next

        # insere no início (mais eficiente)
        new_node = Node(case_id)
        new_node.next = self.postings_head
        self.postings_head = new_node

    def get_postings(self) -> list[str]:
        """Retorna lista de case_id presentes na lista ligada (ordem LIFO)."""

        result = []
        current = self.postings_head
        while current is not None:
            result.append(current.case_id)
            current = current.next

        return result


def build_index_from_cases(path: str | Path) -> dict[str, InvertedIndex]:
    """Carrega casos com preprocess_cases e constrói um índice invertido.

    Usa case["case_id"] como identificador dos documentos.
    """

    cases = preprocess_cases(path=path) # dicionário com as listas de dados pré processados e o objeto df

    inverted_index = {}

    for case in cases:
        # assume que a coluna case_id existe no CSV e foi preservada
        case_id = case.get("case_id")
        if case_id is None:
            # pular casos sem identificador explícito
            continue

        for token in case["retrieval_tokens"]:
            if token not in inverted_index:
                inverted_index[token] = InvertedIndex(token)

            inverted_index[token].add_posting(case_id)

    return inverted_index


def example_usage():
    path = Path("sample/cases.csv")
    index = build_index_from_cases(path)

    # exemplo: ver postings para 'lipase'
    if "lipase" in index:
        print(index["lipase"].get_postings())
    else:
        print([])


if __name__ == "__main__":
    example_usage()