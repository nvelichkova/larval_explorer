"""Upstream column names are spelled only in core/schema.py (CLAUDE.md §4)."""

import ast
from pathlib import Path

from larval_explorer.core import schema

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "larval_explorer"

# Upstream spellings that differ from the canonical name. Channel names are
# identical in both vocabularies, and short generic words ("type", "end") would
# match ordinary strings, so neither is checked.
ALL_SCHEMAS = [value for value in vars(schema).values() if isinstance(value, schema.TableSchema)]
UPSTREAM_ONLY = {
    upstream
    for table in ALL_SCHEMAS
    for canonical, upstream in table.columns.items()
    if upstream != canonical and (len(upstream) > 5 or not upstream.islower())
}


def test_upstream_only_names_are_what_we_expect():
    assert {"animal ID", "tempo_inicial (s)", "estado", "estado_filtrado", "dX", "dY", "duracao (s)"} <= UPSTREAM_ONLY


def test_no_upstream_column_literals_outside_schema():
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path == PACKAGE_ROOT / "core" / "schema.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in UPSTREAM_ONLY:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} {node.value!r}")
    assert not offenders, offenders
