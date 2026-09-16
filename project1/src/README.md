## Instalação

### Pré-requisitos

- Python 3.10 ou superior;
- `pip` disponível no ambiente;
- navegador moderno, apenas para a interface web local.

Na raiz de `project1`, crie e ative um ambiente virtual.

No Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

As dependências declaradas incluem Pandas, NumPy, scikit-learn, NetworkX, Matplotlib e pytest. O servidor da interface local utiliza a biblioteca padrão do Python e a visualização no navegador usa HTML, CSS, JavaScript e SVG, sem exigir um framework web adicional.

## Execução

Todos os comandos abaixo devem ser executados na raiz de `project1`.

### Gerar o grafo

```bash
python scripts/export_knowledge_graph.py
```

O script lê `data/raw/cases.csv`, carrega os arquivos de `vocabularies/` e atualiza `data/processed/nodes.csv` e `data/processed/edges.csv`.

### Gerar uma visualização PNG

Para visualizar o primeiro caso disponível:

```bash
python scripts/visualize_knowledge_graph.py
```

Para selecionar um caso e um arquivo de saída:

```bash
python scripts/visualize_knowledge_graph.py --case-id PMC5137649_01 --output data/processed/case_graph.png
```

Os tipos de nós são diferenciados por cor e as arestas exibem os nomes das relações. Consulte também `VISUALIZATION.md`.

### Abrir a interface web local

```bash
python scripts/serve_graph_interface.py
```

Depois, acesse `http://127.0.0.1:8000`. Uma porta diferente pode ser informada com `--port`. A página permite selecionar um `case_id`, visualizar nós por tipo e mostrar ou ocultar rótulos das relações. A interface na parte inferior da página, permite visualização dos textos originais ligados a cada `case_id`, o texto normalizado e os tokens vinculados, além da visualização das stopwords filtradas e da matriz TF-IDF (reduzida). A interface também permite realizar buscas utilizando boolean retrieval ou boolean ranked.

```bash
python scripts/serve_graph_interface.py --port 8080
```

Para encerrar o servidor, pressione `Ctrl+C`. Consulte também `LOCAL_INTERFACE.md`.

### Executar os testes

```bash
python -m pytest
```

Os testes em `tests/` cobrem pré-processamento, entidades, relações, regras de relação, esquema, construção do grafo, medições, busca ranqueada, visualização e interface web. Existe ainda um teste específico do carregamento de vocabulários em `src/vocab/test_vocabulary.py`.
