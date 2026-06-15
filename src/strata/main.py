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
from .utils.projects import find_project_root

DEFAULT_CHECKPOINT_ROOT = find_project_root() / ".strata" / "checkpoint"


class Main:
    def __init__(self, checkpoint_root: Union[str, pathlib.Path] = DEFAULT_CHECKPOINT_ROOT):
        # The checkpoint store is the single artifact home: every doc is parsed from
        # its copy under the root, and an existing root is inherited on startup.
        self._docs: dict[str, MinerUDocument] = {}
        self._store = Checkpoint.new(checkpoint_root)
        self._restore()

    def open(
        self,
        doc_path: Union[str, pathlib.Path],
        doc_id: Optional[str] = None,
        provider: str = ProviderType.MINERU,
    ) -> str:
        """Back the artifact up under the checkpoint, then parse from that copy.

        Returns the doc_id. Re-opening the same id re-backs-up and re-parses.
        """
        resolved_id = doc_id or get_analyzer(provider)(doc_path).default_doc_id
        ckpt = self._store.doc(resolved_id).save(doc_path, doc_id=resolved_id, provider=provider)
        self._load(ckpt.artifact, resolved_id, provider)
        return resolved_id

    def _load(
        self,
        artifact: Union[str, pathlib.Path],
        doc_id: str,
        provider: str,
    ) -> None:
        # Parse a checkpoint artifact into an in-memory index, cached under doc_id.
        # Shared by open (fresh backup) and restore (existing backup).
        records = get_analyzer(provider)(artifact).analyze(doc_id)
        self._docs[doc_id] = MinerUDocument(doc_id, records)

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
