"""Stratified sampling over a flat ChunkRecord index.

A provider-agnostic consumer parallel to Document: holds one document's records
and draws representative subsets -- either whole pages or individual chunks. Pure
selection over the record contract, with no artifact I/O, so (unlike Document) it
needs no artifact_root. Sampling by region kind / label is expressed by passing a
`key` function, keeping this module free of any provider's label vocabulary.

Two draw modes per unit: without replacement (the default `sample_*`, deduped and
kept in reading order) and with replacement (`*_with_replacement`, a bootstrap
that may repeat picks and comes back in draw order).
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

from .record import ChunkRecord


@dataclass
class PageSample:
    """A sampled page delivered whole: its index plus every record on it in
    reading order. The page is the sampling unit -- nothing inside it is dropped."""
    page_idx : int
    records  : list


def _stratify(records: list[ChunkRecord], key: Optional[Callable[[ChunkRecord], object]]) -> dict:
    # Group records into strata by `key`; key=None is a single global pool. Shared
    # by every draw method -- only the draw + assembly differs between them.
    if key is None:
        return {None: records}
    groups = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    return groups


def _draw_count(total: int, k: Optional[int], fraction: Optional[float], capped: bool = True) -> int:
    # How many to draw from a pool of `total`. At most one of k / fraction is set;
    # neither means take the whole pool (a same-size bootstrap when replacing). A
    # fraction rounds to >=1 so a non-empty pool always yields something. `capped`
    # holds the count to the pool size for without-replacement; replacement lifts
    # it, since a pick can repeat.
    if k is not None and fraction is not None:
        raise ValueError("pass at most one of k / fraction")
    if k is not None:
        return min(k, total) if capped else k
    if fraction is not None:
        n = max(1, round(total * fraction))
        return min(total, n) if capped else n
    return total


class Sampler:

    def __init__(self, records: list[ChunkRecord], doc_id: Optional[str] = None):
        # doc_id is self-described by the records (flatten stamps it on each); accept
        # an override, else read it back rather than make the caller restate it.
        self.doc_id = doc_id if doc_id is not None else (records[0].doc_id if records else None)
        self.records = records

    def sample_pages(self, k: Optional[int] = None, fraction: Optional[float] = None, seed: Optional[int] = None) -> list[PageSample]:
        # Whole-page sampling without replacement: pick a subset of page indices,
        # return each with all its records. Output stays in page order.
        by_page = _stratify(self.records, lambda r: r.page_idx)
        pages = list(by_page)
        chosen = random.Random(seed).sample(pages, _draw_count(len(pages), k, fraction))
        return [PageSample(page_idx=p, records=by_page[p]) for p in sorted(chosen)]

    def sample_chunks(self, k: Optional[int] = None, fraction: Optional[float] = None, key: Optional[Callable[[ChunkRecord], object]] = None, seed: Optional[int] = None) -> list[ChunkRecord]:
        # Chunk sampling without replacement. With `key`, k / fraction apply *per
        # stratum* -- "one chunk per page" is k=1, key=page; "two per kind" is k=2,
        # key=kind. Without `key`, they apply to the whole pool. Output is deduped
        # and kept in reading order.
        rng = random.Random(seed)
        chosen = set()
        for pool in _stratify(self.records, key).values():
            for r in rng.sample(pool, _draw_count(len(pool), k, fraction)):
                chosen.add(r.bbox_id)
        return [r for r in self.records if r.bbox_id in chosen]

    def sample_pages_with_replacement(self, k: Optional[int] = None, fraction: Optional[float] = None, seed: Optional[int] = None) -> list[PageSample]:
        # Whole-page bootstrap: the same page can be drawn more than once. Output is
        # in draw order and may repeat. k / fraction are uncapped (None = pool size).
        by_page = _stratify(self.records, lambda r: r.page_idx)
        pages = list(by_page)
        if not pages:
            return []
        drawn = random.Random(seed).choices(pages, k=_draw_count(len(pages), k, fraction, capped=False))
        return [PageSample(page_idx=p, records=by_page[p]) for p in drawn]

    def sample_chunks_with_replacement(self, k: Optional[int] = None, fraction: Optional[float] = None, key: Optional[Callable[[ChunkRecord], object]] = None, seed: Optional[int] = None) -> list[ChunkRecord]:
        # Chunk bootstrap: within each stratum a record can be picked more than once.
        # Output is in draw order and may repeat, so it skips the dedup / reading-order
        # pass of sample_chunks. k / fraction are uncapped (None = stratum size).
        rng = random.Random(seed)
        drawn = []
        for pool in _stratify(self.records, key).values():
            if not pool:
                continue
            drawn.extend(rng.choices(pool, k=_draw_count(len(pool), k, fraction, capped=False)))
        return drawn

    PAGE_BASED_SAMPLING = {
        sample_pages,
        sample_pages_with_replacement,
    }

    CHUNK_BASED_SAMPLING = {
        sample_chunks,
        sample_chunks_with_replacement,
    }