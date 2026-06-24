"""Stratified sampling over a flat ChunkRecord index.

A provider-agnostic consumer parallel to Document: holds one document's records
and draws representative subsets -- either whole pages or individual chunks. Pure
selection over the record contract, with no artifact I/O, so (unlike Document) it
needs no artifact_root. Sampling by region kind / label is expressed by passing a
`key` function, keeping this module free of any provider's label vocabulary.
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


def _draw_count(total: int, k: Optional[int], fraction: Optional[float]) -> int:
    # How many to draw from a pool of `total`. At most one of k / fraction is set;
    # neither means take the whole pool. A fraction rounds to >=1 so a non-empty
    # pool always yields something, and k is capped at what the pool can give.
    if k is not None and fraction is not None:
        raise ValueError("pass at most one of k / fraction")
    if k is not None:
        return min(k, total)
    if fraction is not None:
        return min(total, max(1, round(total * fraction)))
    return total


class Sampler:
    def __init__(self, records: list[ChunkRecord], doc_id: Optional[str] = None):
        # doc_id is self-described by the records (flatten stamps it on each); accept
        # an override, else read it back rather than make the caller restate it.
        self.doc_id = doc_id if doc_id is not None else (records[0].doc_id if records else None)
        self.records = records

    def sample_pages(self, k: Optional[int] = None, fraction: Optional[float] = None, seed: Optional[int] = None) -> list[PageSample]:
        # Whole-page sampling: pick a subset of page indices, return each with all
        # its records. Output stays in page order regardless of pick order.
        by_page = defaultdict(list)
        for r in self.records:
            by_page[r.page_idx].append(r)
        pages = list(by_page)
        chosen = random.Random(seed).sample(pages, _draw_count(len(pages), k, fraction))
        return [PageSample(page_idx=p, records=by_page[p]) for p in sorted(chosen)]

    def sample_chunks(self, k: Optional[int] = None, fraction: Optional[float] = None, key: Optional[Callable[[ChunkRecord], object]] = None, seed: Optional[int] = None) -> list[ChunkRecord]:
        # Chunk sampling. With `key`, records are grouped into strata and k / fraction
        # apply *per stratum* -- "one chunk per page" is k=1, key=page; "two per kind"
        # is k=2, key=kind. Without `key`, they apply to the whole pool. Output keeps
        # reading order.
        rng = random.Random(seed)
        if key is None:
            strata = {None: self.records}
        else:
            strata = defaultdict(list)
            for r in self.records:
                strata[key(r)].append(r)
        chosen = set()
        for pool in strata.values():
            for r in rng.sample(pool, _draw_count(len(pool), k, fraction)):
                chosen.add(r.bbox_id)
        return [r for r in self.records if r.bbox_id in chosen]
