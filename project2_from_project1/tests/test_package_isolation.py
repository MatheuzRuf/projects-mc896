"""Regression checks for the standalone experimental package."""

import ast
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def test_package_has_no_project1_imports():
    """The experimental package must never load implementation from project1."""

    violations = []
    for path in PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.Import):
                module = ", ".join(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module
            if module and (module == "project1" or module.startswith("project1.")):
                violations.append(f"{path.relative_to(PACKAGE_DIR)}:{node.lineno}: {module}")

    assert not violations, "Unexpected project1 imports:\n" + "\n".join(violations)
