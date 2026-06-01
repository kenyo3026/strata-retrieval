"""Per-document retrieval core over a flat ChunkRecord index.

Holds one document's records in memory and exposes the structural retrieval
operations (PRP section 4.3). Returns rich typed results -- serialization to
dict / JSON / image-content is left to the interface adapters (cli/api/mcp).
"""

import re
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
class GrepMatch:
    bbox_id      : str
    page_idx     : int
    label        : str
    snippet      : str   # window centred on the match
    match_offset : int   # offset of the match within the block content


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


def _grep_snippet(content: str, start: int, end: int, window: int = 30) -> str:
    lo = max(0, start - window)
    hi = min(len(content), end + window)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(content) else ""
    return prefix + " ".join(content[lo:hi].split()) + suffix


class MinerUDocument:
    def __init__(self, doc_id: str, records: list[ChunkRecord]):
        self.doc_id = doc_id
        self.records = records
        self._by_id = {r.bbox_id: r for r in records}
        self._by_page = defaultdict(list)
        self._by_parent = defaultdict(list)   # composite bbox_id -> child SubBlock ids
        self._position = {}                    # bbox_id -> index in reading order
        for i, r in enumerate(records):
            self._by_page[r.page_idx].append(r)
            self._position[r.bbox_id] = i
            if r.parent_bbox_id is not None:
                self._by_parent[r.parent_bbox_id].append(r.bbox_id)

    def read_block(self, bbox_id: str) -> ChunkRecord:
        return self._by_id[bbox_id]

    def read_page(self, page_idx: int) -> list[ChunkRecord]:
        # Full records for the page, in reading order (insertion order is doc order).
        return list(self._by_page.get(page_idx, []))

    def read_block_with_context(self, bbox_id: str, n_prev: int = 1, n_next: int = 1) -> list[ChunkRecord]:
        # The block plus its n_prev/n_next reading-order neighbours, in order.
        pos = self._position[bbox_id]
        lo = max(0, pos - n_prev)
        hi = min(len(self.records), pos + n_next + 1)
        return self.records[lo:hi]

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

    def grep(self, pattern: str, ignore_case: bool = False) -> list[GrepMatch]:
        # Regex search over block content (inline $latex$ is already spliced in).
        # One match (the first) per block; blocks with empty content are skipped.
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        matches = []
        for r in self.records:
            if not r.content:
                continue
            m = regex.search(r.content)
            if m:
                matches.append(
                    GrepMatch(
                        bbox_id=r.bbox_id,
                        page_idx=r.page_idx,
                        label=r.label,
                        snippet=_grep_snippet(r.content, m.start(), m.end()),
                        match_offset=m.start(),
                    )
                )
        return matches

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

    def parent(self, bbox_id: str) -> Optional[str]:
        # Composite parent id, or None for a top-level SubBlock.
        return self._by_id[bbox_id].parent_bbox_id

    def siblings(self, bbox_id: str) -> list[str]:
        # Co-members under the same composite. Top-level blocks have no grouping.
        parent_id = self._by_id[bbox_id].parent_bbox_id
        if parent_id is None:
            return []
        return [bid for bid in self._by_parent[parent_id] if bid != bbox_id]

    def next(self, bbox_id: str) -> Optional[str]:
        pos = self._position[bbox_id] + 1
        return self.records[pos].bbox_id if pos < len(self.records) else None

    def prev(self, bbox_id: str) -> Optional[str]:
        pos = self._position[bbox_id] - 1
        return self.records[pos].bbox_id if pos >= 0 else None
