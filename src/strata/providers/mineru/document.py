"""Per-document retrieval core over a flat ChunkRecord index.

Holds one document's records in memory and exposes the structural retrieval
operations (PRP section 4.3). Returns rich typed results -- serialization to
dict / JSON / image-content is left to the interface adapters (cli/api/mcp).
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

from .chunk import ChunkRecord

_SNIPPET_LEN = 80


@dataclass
class BlockSummary:
    """Compact block listing -- metadata plus a short snippet, no full content."""
    bbox_id  : str
    label    : str
    page_idx : int
    bbox     : Optional[list]
    score    : Optional[float]
    snippet  : str


@dataclass
class PageInfo:
    page_idx        : int
    page_size       : Optional[list]
    counts_by_label : dict
    block_ids       : list


@dataclass
class OutlineEntry:
    """One title in document order. `level` is MinerU's heading level (may be None)."""
    bbox_id : str
    title   : str
    page    : int
    level   : Optional[int]


def _snippet(content: str, length: int = _SNIPPET_LEN) -> str:
    collapsed = " ".join(content.split())
    return collapsed if len(collapsed) <= length else collapsed[:length] + "..."


class MinerUDocument:
    def __init__(self, doc_id: str, records: list[ChunkRecord]):
        self.doc_id = doc_id
        self.records = records
        self._by_id = {r.bbox_id: r for r in records}
        self._by_page = defaultdict(list)
        for r in records:
            self._by_page[r.page_idx].append(r)

    def read_block(self, bbox_id: str) -> ChunkRecord:
        return self._by_id[bbox_id]

    def list_blocks(self, label: Optional[str] = None, page: Optional[int] = None) -> list[BlockSummary]:
        summaries = []
        for r in self.records:
            if label is not None and r.label != label:
                continue
            if page is not None and r.page_idx != page:
                continue
            summaries.append(
                BlockSummary(
                    bbox_id=r.bbox_id,
                    label=r.label,
                    page_idx=r.page_idx,
                    bbox=r.bbox,
                    score=r.score,
                    snippet=_snippet(r.content),
                )
            )
        return summaries

    def outline(self, label_keyword: str = "title") -> list[OutlineEntry]:
        # Titles in document order. Match any label containing "title"
        # (title / doc_title / paragraph_title) to stay robust to MinerU's open vocab.
        return [
            OutlineEntry(bbox_id=r.bbox_id, title=r.content, page=r.page_idx, level=r.level)
            for r in self.records
            if label_keyword in r.label
        ]

    def page_info(self, page_idx: int) -> PageInfo:
        records = self._by_page.get(page_idx, [])
        page_size = records[0].page_size if records else None
        return PageInfo(
            page_idx=page_idx,
            page_size=page_size,
            counts_by_label=dict(Counter(r.label for r in records)),
            block_ids=[r.bbox_id for r in records],
        )
