"""Core orchestrator for strata-retrieval.

Thin façade: manages document handles (parse + cache an index per doc) and
provider selection. The retrieval tools themselves live on the per-document core
(MinerUDocument); Main only owns lifecycle. The cli/api/mcp adapters wrap Main.
"""

import pathlib
from typing import Optional, Union

from .checkpoint import Checkpoint
from .providers.factory import ProviderType, get_analyzer
from .providers.mineru.document import MinerUDocument


class Main:
    def __init__(self, checkpoint_root: Optional[Union[str, pathlib.Path]] = None):
        self._docs: dict[str, MinerUDocument] = {}
        self._store = Checkpoint.new(checkpoint_root) if checkpoint_root else None
        if self._store is not None:
            self._restore()

    def open(
        self,
        doc_path: Union[str, pathlib.Path],
        doc_id: Optional[str] = None,
        provider: str = ProviderType.MINERU,
    ) -> str:
        """Parse a document dir into a flat index and cache it. Returns doc_id."""
        resolved_id = self._load(doc_path, doc_id, provider)
        if self._store is not None:
            self._store.doc(resolved_id).save(doc_path, doc_id=resolved_id, provider=provider)
        return resolved_id

    def _load(
        self,
        doc_path: Union[str, pathlib.Path],
        doc_id: Optional[str],
        provider: str,
    ) -> str:
        # Parse + cache into memory only -- no persistence (restore reuses this).
        analyzer = get_analyzer(provider)(doc_path)
        resolved_id = doc_id or analyzer.default_doc_id
        records = analyzer.analyze(resolved_id)
        self._docs[resolved_id] = MinerUDocument(resolved_id, records)
        return resolved_id

    def _restore(self) -> None:
        # Re-open every doc already backed up under the checkpoint root.
        for doc_id in self._store.doc_ids():
            ckpt = self._store.doc(doc_id)
            self._load(ckpt.artifact, doc_id, ckpt.meta()["provider"])

    def close(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)

    def doc(self, doc_id: str) -> MinerUDocument:
        return self._docs[doc_id]

    def doc_summaries(self) -> list:
        # Lightweight overview of open docs (id + counts, no content).
        return [d.summary() for d in self._docs.values()]
