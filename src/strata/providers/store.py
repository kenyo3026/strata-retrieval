"""Persist a flat ChunkRecord index as one JSONL per document.

Storage is intentionally separable from the record definition: per PRP section 10
the backend may later move to SQLite FTS5 / a vector store while ChunkRecord stays
fixed. Keep that swap confined to this module.
"""

import json
import pathlib
from dataclasses import asdict

from .record import ChunkRecord


def write_jsonl(records: list[ChunkRecord], path: pathlib.Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False))
            f.write("\n")


def read_jsonl(path: pathlib.Path) -> list[ChunkRecord]:
    with open(path, encoding="utf-8") as f:
        return [ChunkRecord(**json.loads(line)) for line in f if line.strip()]
