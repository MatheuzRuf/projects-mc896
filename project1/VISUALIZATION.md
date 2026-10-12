# Visualizacao do Knowledge Graph

A visualizacao usa Python, NetworkX e Matplotlib. Ela carrega os arquivos
`output/nodes.csv` e `output/edges.csv`, seleciona um caso e salva o grafo em
`output/graph.png`.

## Executar localmente

Na raiz do repositório, instale as dependencias e o pacote:

```bash
pip install -r requirements.txt
pip install -e .
```

Gere a visualizacao do primeiro caso:

```bash
python -m project1.scripts.visualize_knowledge_graph
```

Ou escolha um caso e outro arquivo de saida:

```bash
python -m project1.scripts.visualize_knowledge_graph --case-id PMC5137649_01 --output output/case_graph.png
```

Os tipos de nos sao diferenciados por cor e cada aresta mostra o nome da
relacao.
