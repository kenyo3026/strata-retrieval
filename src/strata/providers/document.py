"""Per-document retrieval core over a flat ChunkRecord index.

Holds one document's records in memory and exposes the structural retrieval
operations (PRP section 4.3). Returns rich typed results -- serialization to
dict / JSON / image-content is left to the interface adapters (cli/api/mcp).
"""

import base64
import mimetypes
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Optional, Union

from .record import ChunkRecord

_SNIPPET_LEN = 80


@dataclass
class DocSummary:
    """Lightweight per-document overview -- counts only, no block content."""
    doc_id   : Optional[str]
    n_pages  : int
    n_blocks : int


@dataclass
class BlockSummary:
    """Compact block listing -- metadata plus a short snippet, no full content."""
    bbox_id   : str
    label     : str
    page_idx  : int
    bbox      : Optional[list]
    page_size : Optional[list]
    norm_bbox : Optional[list]   # bbox as page-relative [0..1] fractions
    score     : Optional[float]
    snippet   : str


@dataclass(frozen=True)
class RegionKind:
    """The small closed vocabulary a page region collapses to (MinerU's open
    `label` -> one of these), driving the per-kind delivery payload."""
    TEXT     : str = "text"
    EQUATION : str = "equation"
    TABLE    : str = "table"
    IMAGE    : str = "image"


@dataclass
class TextRegion:
    """Any inline-string region (text / title / equation / table): the payload
    lives in `content`, and `kind` says how to read it -- plain text (with inline
    `$latex$` spliced in), `$$...$$` LaTeX, or an html table string."""
    bbox_id : str
    kind    : str
    label   : str
    bbox    : Optional[list]
    content : Optional[str]


@dataclass
class ImageRegion:
    """Image / chart as a zero-I/O placeholder: the reference plus its caption
    (folded in from a sibling SubBlock), no bytes. The caption lets an LLM decide
    whether to fetch the image without blindly loading it."""
    bbox_id    : str
    kind       : str
    label      : str
    bbox       : Optional[list]
    image_path : Optional[str]
    caption    : Optional[str] = None
    content    : Optional[str] = None   # base64 data uri when embed_images=True


@dataclass
class PagePayload:
    """A whole page as the delivery unit (rag-as-book): a page header plus its
    regions in reading order, each shaped per its kind."""
    doc_id    : Optional[str]
    page_idx  : int
    page_size : Optional[list]
    regions   : list


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


def _norm_bbox(bbox: Optional[list], page_size: Optional[list]) -> Optional[list]:
    # bbox [x0,y0,x1,y1] -> page-relative [0..1] fractions; None if either is missing.
    if not bbox or not page_size:
        return None
    w, h = page_size
    if not w or not h:
        return None
    return [round(bbox[0] / w, 3), round(bbox[1] / h, 3), round(bbox[2] / w, 3), round(bbox[3] / h, 3)]


def _kind_of(record: ChunkRecord) -> str:
    # Drive kind off the payload, not the label string: an interline equation is
    # content; an html means table; else an image_path means image -- which also
    # folds the "table misdetected as image, no html" case straight into image.
    if "equation" in record.label:
        return RegionKind.EQUATION
    elif record.html:
        return RegionKind.TABLE
    elif record.image_path:
        return RegionKind.IMAGE
    return RegionKind.TEXT


def _region(record: ChunkRecord, caption: Optional[str] = None):
    # Shape a flat record into its per-kind delivery region. caption is folded in
    # from a sibling SubBlock for image/chart (image-byte embedding comes later).
    kind = _kind_of(record)
    if kind == RegionKind.IMAGE:
        return ImageRegion(bbox_id=record.bbox_id, kind=kind, label=record.label, bbox=record.bbox, image_path=record.image_path, caption=caption)
    content = record.html if kind == RegionKind.TABLE else record.content
    return TextRegion(bbox_id=record.bbox_id, kind=kind, label=record.label, bbox=record.bbox, content=content)


def _assemble_regions(records: list) -> list:
    # Build a page's regions, folding each image/chart's sibling caption(s) into
    # the image region (same composite parent) and dropping them as standalone.
    siblings = defaultdict(list)
    for r in records:
        siblings[r.parent_bbox_id].append(r)

    folded = {}      # image bbox_id -> joined caption text
    consumed = set()  # caption bbox_ids folded away, so they aren't re-emitted
    for r in records:
        if _kind_of(r) != RegionKind.IMAGE or r.parent_bbox_id is None:
            continue
        caps = [s for s in siblings[r.parent_bbox_id] if "caption" in s.label and s.content]
        if caps:
            folded[r.bbox_id] = " ".join(s.content for s in caps)
            consumed.update(s.bbox_id for s in caps)

    return [_region(r, caption=folded.get(r.bbox_id)) for r in records if r.bbox_id not in consumed]


def _grep_snippet(content: str, start: int, end: int, window: int = 30) -> str:
    lo = max(0, start - window)
    hi = min(len(content), end + window)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(content) else ""
    return prefix + " ".join(content[lo:hi].split()) + suffix


class Document:
    def __init__(self, records: list[ChunkRecord], artifact_root: Union[str, pathlib.Path], doc_id: Optional[str] = None):
        # doc_id is self-described by the records (flatten stamps it on each); accept
        # an override, else read it back rather than make the caller restate it.
        self.doc_id = doc_id if doc_id is not None else (records[0].doc_id if records else None)
        self.records = records
        self.artifact_root = pathlib.Path(artifact_root)   # resolves image_path for embedding
        self._by_id = {r.bbox_id: r for r in self.records}
        self._by_page = defaultdict(list)
        self._by_parent = defaultdict(list)   # composite bbox_id -> child SubBlock ids
        self._position = {}                    # bbox_id -> index in reading order
        for i, r in enumerate(self.records):
            self._by_page[r.page_idx].append(r)
            self._position[r.bbox_id] = i
            if r.parent_bbox_id is not None:
                self._by_parent[r.parent_bbox_id].append(r.bbox_id)

    def summary(self) -> DocSummary:
        return DocSummary(
            doc_id=self.doc_id,
            n_pages=len(self._by_page),
            n_blocks=len(self.records),
        )

    def read_block(self, bbox_id: str) -> ChunkRecord:
        return self._by_id[bbox_id]

    def read_page(self, page_idx: int, embed_images: bool = False) -> PagePayload:
        # The whole page as one delivery unit: page header + per-kind regions in
        # reading order. With embed_images, image regions also inline the image
        # bytes as a base64 data uri; otherwise they stay a zero-I/O reference.
        records = self._by_page.get(page_idx, [])
        regions = _assemble_regions(records)
        if embed_images:
            regions = [self._embed(r) if isinstance(r, ImageRegion) else r for r in regions]
        return PagePayload(
            doc_id=self.doc_id,
            page_idx=page_idx,
            page_size=records[0].page_size if records else None,
            regions=regions,
        )

    def _embed(self, region: ImageRegion) -> ImageRegion:
        # Load the referenced image off the artifact and inline it as a data uri.
        if not region.image_path:
            return region
        path = self.artifact_root / "images" / pathlib.Path(region.image_path).name
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return replace(region, content=f"data:{mime};base64,{data}")

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
                    page_size=r.page_size,
                    norm_bbox=_norm_bbox(r.bbox, r.page_size),
                    score=r.score,
                    snippet=_snippet(r.content),
                )
            )
        return summaries

    def grep(self, pattern: str, ignore_case: bool = False, limit: Optional[int] = None) -> list[GrepMatch]:
        # Regex search over block content (inline $latex$ is already spliced in).
        # One match (the first) per block; blocks with empty content are skipped.
        # limit caps the number of matched blocks returned (None = all).
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        matches = []
        for r in self.records:
            if limit is not None and len(matches) >= limit:
                break
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
