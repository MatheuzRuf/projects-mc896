# Projetos MC896

Projetos de mc896 2s2026.

Integrantes:

Matheus Rufino da Silva - RA 221756

Gabriel Thomas Dias de Freitas - RA 235071

Daniel de Sousa Cipriano - RA 233228

Eduardo dos Santos Neres - RA 188732

Diogo Barros de Paula - RA 242545

## Estrutura e instalação

```text
.
├── pyproject.toml     # define project1 e project2 como pacotes
├── requirements.txt
├── project1/
└── project2/
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Imports entre projetos usam o nome do pacote, por exemplo:

```python
from project1.src.retrieval.inverted_index import build_index_from_cases
from project1.paths import CASES_PATH
```

Scripts: `python -m project1.scripts.export_knowledge_graph` (de qualquer pasta). Testes: `python -m pytest`.

