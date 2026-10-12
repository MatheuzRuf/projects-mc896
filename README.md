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
├── pyproject.toml     # define project1, project2 e project2_from_project1 como pacotes
├── requirements.txt
├── project1/
├── project2/
└── project2_from_project1/
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

Scripts legados: `python -m project1.scripts.export_knowledge_graph` (de qualquer pasta).

O experimento independente fica em `project2_from_project1` e pode ser validado sem importar implementações de `project1`:

```bash
python -m pytest project2_from_project1/tests project2_from_project1/src/vocab
```

Para executar todas as suítes configuradas no monorepo: `python -m pytest`.

