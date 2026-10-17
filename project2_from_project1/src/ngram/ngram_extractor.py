from collections import Counter
import json

# função de tokenização aproveitada do projeto 1, sem remover stopwords (usa regex)
from project2_from_project1.src.preprocessing.text import tokenize

def generate_ngrams(tokens: list[str]) -> tuple[Counter, Counter, Counter]:
    """gera dicionários de frequência para unigramas, bigramas e trigramas"""

    # unigramas: conta cada token isoladamente
    unigrams = Counter(tokens)
    
    # bigramas: agrupa (token_atual, token_seguinte)
    bigrams = Counter(zip(tokens, tokens[1:]))
    
    # trigramas: agrupa (token_atual, token_seguinte, token_mais_a_frente)
    trigrams = Counter(zip(tokens, tokens[1:], tokens[2:]))
    
    return unigrams, bigrams, trigrams

# OBS: JSONL é um formato de arquivo onde cada linha é um objeto JSON válido
def process_corpus_ngrams(jsonl_path: str) -> tuple[Counter, Counter, Counter]:
    """
    lê um arquivo JSONL (ex: fold-XX-train.jsonl), extrai os textos 
    de contexto e acumula as frequências globais de n-gramas
    """
    global_unigrams = Counter()
    global_bigrams = Counter()
    global_trigrams = Counter()
    
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip(): # ignora linhas vazias
                continue
                
            data = json.loads(line)
            
            text = data.get("context_text", "") # adaptar o nome da chave quando possuir o arquivo JSONL real
            if not text:
                continue
            
            tokens = tokenize(text)
            uni, bi, tri = generate_ngrams(tokens)
            
            global_unigrams.update(uni)
            global_bigrams.update(bi)
            global_trigrams.update(tri)
            
    return global_unigrams, global_bigrams, global_trigrams

if __name__ == "__main__":
    # teste
    text = "pneumonia secondary to aspiration"
    tokens = tokenize(text)
    uni, bi, tri = generate_ngrams(tokens)
    
    print("--- Teste de Extração de N-grams ---")
    print(f"Texto original: '{text}'\n")
    print(f"Tokens: {tokens}\n")
    
    print("Unigramas Encontrados:")
    for gram, count in uni.items():
        print(f"  {gram}: {count}")
        
    print("\nBigramas Encontrados:")
    for gram, count in bi.items():
        print(f"  {gram}: {count}")
        
    print("\nTrigramas Encontrados:")
    for gram, count in tri.items():
        print(f"  {gram}: {count}")