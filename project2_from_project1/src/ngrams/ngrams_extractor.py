from collections import Counter
from collections.abc import Iterable, Sequence
import json
from typing import Literal, overload

# função de tokenização aproveitada do projeto 1, sem remover stopwords (usa regex)
from project2_from_project1.src.preprocessing.text import tokenize, normalize_text

NgramCounts = tuple[
    Counter[str],
    Counter[tuple[str, str]],
    Counter[tuple[str, str, str]],
]
NgramFeature = tuple[int, tuple[str, ...]]

# janela de contexto (em palavras e em caracteres) usada quando só uma das menções tem texto
CONTEXT_WINDOW_WORDS = 4
CONTEXT_WINDOW_CHARS = 200

def generate_ngrams(tokens: list[str]) -> tuple[Counter, Counter, Counter]:
    """
    gera dicionários de frequência para unigramas, bigramas e trigramas"
    
    params:
        tokens: lista de tokens
    """
    # unigramas: conta cada token isoladamente
    unigrams = Counter(tokens)
    
    # bigramas: agrupa (token_atual, token_seguinte)
    bigrams = Counter(zip(tokens, tokens[1:]))
    
    # trigramas: agrupa (token_atual, token_seguinte, token_mais_a_frente)
    trigrams = Counter(zip(tokens, tokens[1:], tokens[2:]))
    
    return unigrams, bigrams, trigrams

def build_ngram_vocabulary(ngram_counters: Iterable[NgramCounts]) -> tuple[NgramFeature, ...]:
    """Cria um esquema determinístico de características para contadores de n-gramas."""
    features: set[NgramFeature] = set()

    for unigrams, bigrams, trigrams in ngram_counters:
        features.update((1, (token,)) for token in unigrams)
        features.update((2, bigram) for bigram in bigrams)
        features.update((3, trigram) for trigram in trigrams)

    return tuple(sorted(features))

def _iter_ngram_feature_counts(ngram_counts: NgramCounts):
    unigrams, bigrams, trigrams = ngram_counts
    yield from (((1, (token,)), count) for token, count in unigrams.items())
    yield from (((2, gram), count) for gram, count in bigrams.items())
    yield from (((3, gram), count) for gram, count in trigrams.items())

@overload
def vectorize_ngrams(
    ngram_counts: NgramCounts,
    vocabulary: Sequence[NgramFeature],
    *,
    sparse: Literal[False] = False,
) -> list[int]: ...

@overload
def vectorize_ngrams(
    ngram_counts: NgramCounts,
    vocabulary: Sequence[NgramFeature],
    *,
    sparse: Literal[True],
) -> dict[int, int]: ...

def vectorize_ngrams(
    ngram_counts: NgramCounts,
    vocabulary: Sequence[NgramFeature],
    *,
    sparse: bool = False,
) -> list[int] | dict[int, int]:
    """Retorna frequências na ordem do vocabulário; opcionalmente, apenas colunas não zero."""
    feature_positions = {feature: index for index, feature in enumerate(vocabulary)}
    if sparse:
        return {
            position: count
            for feature, count in _iter_ngram_feature_counts(ngram_counts)
            if (position := feature_positions.get(feature)) is not None and count != 0
        }

    vector = [0] * len(vocabulary)
    for feature, count in _iter_ngram_feature_counts(ngram_counts):
        position = feature_positions.get(feature)
        if position is not None:
            vector[position] = count
    return vector

def _window_tokens(full_text: str, start: int, end: int) -> list[str]:
    # palavras antes e depois da menção, para candidatos cujo outro nó não tem texto (como Patient)
    left = full_text[max(0, start - CONTEXT_WINDOW_CHARS):start].split()[-CONTEXT_WINDOW_WORDS:]
    right = full_text[end:end + CONTEXT_WINDOW_CHARS].split()[:CONTEXT_WINDOW_WORDS]
    normalized = normalize_text(" ".join([*left, full_text[start:end], *right]))
    return tokenize(normalized)

def extract_candidate_tokens(candidate: dict, full_text: str, mentions_by_id: dict) -> list[str]:
    """
    retorna os tokens de contexto de um candidato de relação

    params:
        candidate: registro do candidato (source_id e target_id)
        full_text: texto bruto do documento
        mentions_by_id: menções do documento indexadas por mention_id
    """
    # encontra as posições exatas (spans) da origem e do destino no texto
    # (o corpus armazena as menções com seus spans, então podemos recuperar o texto exato)
    # span tem o formato [start, end] correspondente aos caracteres no full_text
    spans = sorted(
        tuple(mentions_by_id[candidate[key]]["spans"][0])
        for key in ("source_id", "target_id")
        if candidate[key] in mentions_by_id
    )

    # nós lógicos (como Patient sem texto base) não possuem menção; sem nenhuma menção com texto não há contexto
    if not spans:
        return []

    # candidato com nó lógico (como Patient sem texto base): usa a janela ao redor da única menção com texto
    if len(spans) == 1:
        return _window_tokens(full_text, *spans[0])

    # determina qual entidade vem primeiro no texto para pegar o contexto entre elas
    first, second = spans
    if first[1] <= second[0]:
        normalized = normalize_text(full_text[first[1]:second[0]].strip())
        tokens = tokenize(normalized)
        if tokens:
            return tokens

    # se as entidades estiverem coladas ou sobrepostas, o contexto é vazio; usa o trecho que cobre as duas
    normalized = normalize_text(full_text[first[0]:max(first[1], second[1])])
    return tokenize(normalized)

# OBS: JSONL é um formato de arquivo onde cada linha é um objeto JSON válido
def process_corpus_ngrams(fold_path: str, corpus_path: str) -> tuple[Counter, Counter, Counter]:
    """
    lê um arquivo JSONL de treino ou validação, extrai os n-gramas 
    e retorna contadores globais de unigramas, bigramas e trigramas

    params:
        fold_path: caminho para o arquivo JSONL do fold (treino ou validação)
        corpus_path: caminho do arquivo JSONL que possui o texto bruto
    """
    global_unigrams = Counter()
    global_bigrams = Counter()
    global_trigrams = Counter()

    documents_by_id = {} # armazena os documentos do corpus para acesso rápido pelo document_id
    mentions_by_document = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            doc = json.loads(line)
            documents_by_id[doc["document_id"]] = doc
            mentions_by_document[doc["document_id"]] = {m["mention_id"]: m for m in doc["mentions"]}
    
    with open(fold_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): 
                continue
                
            candidate = json.loads(line) # cada linha é um candidato de relação
            doc_id = candidate["document_id"] # recupera o document_id do candidato
            
            # usa a "Chave Estrangeira" para recuperar o documento completo
            document = documents_by_id[doc_id]
            full_text = document["text"] # texto bruto

            tokens = extract_candidate_tokens(candidate, full_text, mentions_by_document[doc_id])
            
            # evita extrair n-gramas se não houver tokens suficientes
            if not tokens:
                continue

            uni, bi, tri = generate_ngrams(tokens)
            
            global_unigrams.update(uni)
            global_bigrams.update(bi)
            global_trigrams.update(tri)
            
    return global_unigrams, global_bigrams, global_trigrams

if __name__ == "__main__":
    # teste
    corpus_path = "project2_from_project1/data/secure/mcrc.jsonl"
    fold_path = "project2_from_project1/artifacts/first-milestone/frozen-dataset/fold-01-train.jsonl"

    unigrams, bigrams, trigrams = process_corpus_ngrams(fold_path, corpus_path)
    print("Unigrams:", unigrams.most_common(10))
    print("Bigrams:", bigrams.most_common(10))
    print("Trigrams:", trigrams.most_common(10))