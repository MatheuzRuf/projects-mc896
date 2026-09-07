import csv
from pathlib import Path

from .vocabulary import VocabularyEntry


def load_vocabulary(path: str | Path) -> list[VocabularyEntry]:
    """
    Carrega um vocabulário CSV.

    Formato esperado:

        term,aliases,type,source,code

    Onde aliases são separados por '|'.
    """

    path = Path(path)

    entries = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            aliases = tuple(
                alias.strip()
                for alias in row["aliases"].split("|")
                if alias.strip()
            )

            entries.append(
                VocabularyEntry(
                    term=row["term"].strip(),
                    aliases=aliases,
                    type=row["type"].strip(),
                    source=row.get("source", "manual").strip(),
                    code=row.get("code", "").strip(),
                )
            )

    return entries