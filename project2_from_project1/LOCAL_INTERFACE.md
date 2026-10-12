# Interface local do Knowledge Graph

A interface usa somente a biblioteca padrao do Python no servidor e SVG no
navegador. Nao e necessario instalar Flask, Django ou dependencias de
JavaScript.

## Executar

Na raiz do repositório (após `pip install -e .`), execute:

```bash
python -m project1.scripts.serve_graph_interface
```

Depois abra no navegador:

```text
http://127.0.0.1:8000
```

Para usar outra porta:

```bash
python -m project1.scripts.serve_graph_interface --port 8080
```

Para encerrar o servidor, pressione `Ctrl+C` no terminal.
