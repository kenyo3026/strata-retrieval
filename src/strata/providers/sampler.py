"""Stratified sampling over a flat ChunkRecord index.

A provider-agnostic consumer parallel to Document: holds one document's records
and draws representative subsets -- whole pages, whole sections, or individual
chunks. Pure selection over the record contract, with no artifact I/O, so (unlike
Document) it needs no artifact_root. Sampling by region kind / label is expressed
by passing a `key` function, keeping this module free of any provider's label
vocabulary.

Every mode reduces to the same pipeline: build Units (a chunk is a one-record unit,
a page / section is a many-record unit), then draw among them. Two draw modes per
unit: without replacement (the default `sample_*`, deduped and kept in reading
order) and with replacement (`*_with_replacement`, a bootstrap that may repeat picks
and comes back in draw order).
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

from .record import ChunkRecord, iter_sections, iter_sections_by_level


# Sentinel for sample_sections' `level`: distinguishes "not passed" (keep the legacy
# top-level disjoint projection) from an explicit level=None ("all levels", nested).
_TOP_LEVEL_SECTIONS = object()


@dataclass
class Unit:
    """A group of records drawn atomically -- the sampling unit every mode reduces to.
    `key` is its identity (a chunk's bbox_id, a page index, a section's title bbox_id);
    `records` are its members in reading order. A chunk is a one-record unit."""
    key     : object
    records : list


@dataclass
class PageSample:
    """A sampled page delivered whole: its index plus every record on it in
    reading order. The page is the sampling unit -- nothing inside it is dropped."""
    page_idx : int
    records  : list


@dataclass
class SectionSample:
    """A sampled section delivered whole: its key (the title's bbox_id, or the first
    block for the leading title-less run) plus every record in it in reading order.
    The section is the sampling unit -- its whole subtree is folded in."""
    key     : str
    records : list


def _stratify(items: list, key: Optional[Callable[[object], object]]) -> dict:
    # Group items into strata by `key`; key=None is a single global pool. Shared by
    # unitizing (records -> pages) and drawing (units -> strata) -- only the items differ.
    if key is None:
        return {None: items}
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
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


def _match(record: ChunkRecord, where: dict) -> bool:
    # A record passes when every attr satisfies its condition: membership for a
    # list/set value, equality otherwise.
    for attr, cond in where.items():
        val = getattr(record, attr)
        ok = val in cond if isinstance(cond, (list, set)) else val == cond
        if not ok:
            return False
    return True


def _draw(units: list[Unit], k: Optional[int] = None, fraction: Optional[float] = None,
          key: Optional[Callable[[Unit], object]] = None, with_replacement: bool = False,
          where: Optional[dict] = None, seed: Optional[int] = None) -> list[Unit]:
    # The one draw shared by every mode. `where` filters each unit's records *after* the
    # unit is built (so a section's tree is intact first), dropping units left empty --
    # for a one-record chunk unit that is plain eligibility, for a page / section it keeps
    # only the matching content. Then stratify by `key` (None = one global pool), and per
    # stratum k / fraction apply. Without replacement: a capped subset, returned in the
    # units' original (document) order. With replacement: an uncapped, possibly-repeating
    # bootstrap, returned in draw order.
    if where is not None:
        units = [Unit(u.key, kept) for u in units if (kept := [r for r in u.records if _match(r, where)])]
    rng = random.Random(seed)
    strata = _stratify(units, key)

    if with_replacement:
        drawn: list[Unit] = []
        for pool in strata.values():
            if pool:
                drawn.extend(rng.choices(pool, k=_draw_count(len(pool), k, fraction, capped=False)))
        return drawn

    chosen = set()
    for pool in strata.values():
        for unit in rng.sample(pool, _draw_count(len(pool), k, fraction)):
            chosen.add(id(unit))

    return [u for u in units if id(u) in chosen]


class Sampler:

    def __init__(self, records: list[ChunkRecord], doc_id: Optional[str] = None):
        # doc_id is self-described by the records (flatten stamps it on each); accept
        # an override, else read it back rather than make the caller restate it.
        self.doc_id = doc_id if doc_id is not None else (records[0].doc_id if records else None)
        self.records = records

    def _chunk_units(self) -> list[Unit]:
        # Each record is its own unit; stratification / draw happen over these.
        return [Unit(r.bbox_id, [r]) for r in self.records]

    def _page_units(self) -> list[Unit]:
        # One unit per page, in page order, holding the page's records.
        by_page = _stratify(self.records, lambda r: r.page_idx)
        return [Unit(p, by_page[p]) for p in sorted(by_page)]

    def _section_units(self, level=_TOP_LEVEL_SECTIONS) -> list[Unit]:
        # One unit per section, keyed by the section's title bbox_id. `level` chooses the
        # projection: omitted keeps the top-level disjoint tiling; given (None=all levels,
        # int / list[int]=those levels) switches to the by-level projection, whose
        # sections nest and may overlap.
        sections = iter_sections(self.records) if level is _TOP_LEVEL_SECTIONS else iter_sections_by_level(self.records, level)
        return [Unit(sec[0].bbox_id, sec) for sec in sections]

    def sample_chunks(self, k: Optional[int] = None, fraction: Optional[float] = None, key: Optional[Callable[[ChunkRecord], object]] = None, where: Optional[dict] = None, seed: Optional[int] = None) -> list[ChunkRecord]:
        # Chunk sampling without replacement. With `key`, k / fraction apply *per
        # stratum* -- "one chunk per page" is k=1, key=page; "two per kind" is k=2,
        # key=kind. Without `key`, they apply to the whole pool. `where` keeps only
        # matching records as eligible. Output is deduped and kept in reading order.
        ukey = (lambda u: key(u.records[0])) if key else None
        drawn = _draw(self._chunk_units(), k, fraction, ukey, where=where, seed=seed)
        return [u.records[0] for u in drawn]

    def sample_pages(self, k: Optional[int] = None, fraction: Optional[float] = None, where: Optional[dict] = None, seed: Optional[int] = None) -> list[PageSample]:
        # Whole-page sampling without replacement: pick a subset of pages, return each
        # with all its records. `where` keeps only matching records within each page,
        # dropping pages left empty. Output stays in page order.
        drawn = _draw(self._page_units(), k, fraction, where=where, seed=seed)
        return [PageSample(page_idx=u.key, records=u.records) for u in drawn]

    def sample_sections(self, k: Optional[int] = None, fraction: Optional[float] = None, level=_TOP_LEVEL_SECTIONS, where: Optional[dict] = None, seed: Optional[int] = None) -> list[SectionSample]:
        # Whole-section sampling without replacement: project the records into sections
        # (each a title plus its subtree), pick a subset, and return each whole. Output
        # stays in document order. `level` selects the projection (see _section_units).
        # `where` keeps only matching records within each section (the tree is built from
        # the full records first, so filtering content never breaks sectioning), dropping
        # sections left empty.
        drawn = _draw(self._section_units(level), k, fraction, where=where, seed=seed)
        return [SectionSample(key=u.key, records=u.records) for u in drawn]

    def sample_chunks_with_replacement(self, k: Optional[int] = None, fraction: Optional[float] = None, key: Optional[Callable[[ChunkRecord], object]] = None, where: Optional[dict] = None, seed: Optional[int] = None) -> list[ChunkRecord]:
        # Chunk bootstrap: within each stratum a record can be picked more than once.
        # Output is in draw order and may repeat, so it skips the dedup / reading-order
        # pass of sample_chunks. `where` keeps only matching records as eligible. k /
        # fraction are uncapped (None = stratum size).
        ukey = (lambda u: key(u.records[0])) if key else None
        drawn = _draw(self._chunk_units(), k, fraction, ukey, with_replacement=True, where=where, seed=seed)
        return [u.records[0] for u in drawn]

    def sample_pages_with_replacement(self, k: Optional[int] = None, fraction: Optional[float] = None, where: Optional[dict] = None, seed: Optional[int] = None) -> list[PageSample]:
        # Whole-page bootstrap: the same page can be drawn more than once. Output is in
        # draw order and may repeat. `where` keeps only matching records within each page,
        # dropping pages left empty. k / fraction are uncapped (None = pool size).
        drawn = _draw(self._page_units(), k, fraction, with_replacement=True, where=where, seed=seed)
        return [PageSample(page_idx=u.key, records=u.records) for u in drawn]

    def sample_sections_with_replacement(self, k: Optional[int] = None, fraction: Optional[float] = None, level=_TOP_LEVEL_SECTIONS, where: Optional[dict] = None, seed: Optional[int] = None) -> list[SectionSample]:
        # Whole-section bootstrap: the same section can be drawn more than once. Output
        # is in draw order and may repeat. k / fraction are uncapped (None = pool size).
        # `level` selects the projection and `where` filters section content, as in
        # sample_sections.
        drawn = _draw(self._section_units(level), k, fraction, with_replacement=True, where=where, seed=seed)
        return [SectionSample(key=u.key, records=u.records) for u in drawn]

    PAGE_BASED_SAMPLING = {
        sample_pages,
        sample_pages_with_replacement,
    }

    SECTION_BASED_SAMPLING = {
        sample_sections,
        sample_sections_with_replacement,
    }

    CHUNK_BASED_SAMPLING = {
        sample_chunks,
        sample_chunks_with_replacement,
    }
