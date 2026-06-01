"""Core orchestrator for strata-retrieval.

Thin façade: manages document handles (parse + cache an index per doc) and
provider selection. The retrieval tools themselves live on the per-document core
(MinerUDocument); Main only owns lifecycle. The cli/api/mcp adapters wrap Main.
"""

import pathlib
from typing import Optional, Union

from .providers.mineru.analyzer import MinerUAnalyzer, MinerUArtifact
from .providers.mineru.document import MinerUDocument


class Main:
    def __init__(self):
        self._docs: dict[str, MinerUDocument] = {}

    def open(self, doc_path: Union[str, pathlib.Path], doc_id: Optional[str] = None) -> str:
        """Parse a MinerU `auto` dir into a flat index and cache it. Returns doc_id."""
        artifact = MinerUArtifact.from_dir(pathlib.Path(doc_path))
        resolved_id = doc_id or artifact.basename
        records = MinerUAnalyzer(artifact).analyze(resolved_id)
        self._docs[resolved_id] = MinerUDocument(resolved_id, records)
        return resolved_id

    def close(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)

    def doc(self, doc_id: str) -> MinerUDocument:
        return self._docs[doc_id]
