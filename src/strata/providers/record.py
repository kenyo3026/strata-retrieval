"""The provider-agnostic record contract.

`ChunkRecord` is the normalization boundary: every provider's analyzer flattens
its own raw format into this shape, and every consumer (Document, store, future
sampler) reads only this -- nothing downstream touches a provider's raw structure.
The standard vocabulary (RegionKind / role) that lets consumers stop sniffing a
provider's open label strings will land here too.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkRecord:
    doc_id           : str
    bbox_id          : str
    parent_bbox_id   : Optional[str]  # composite block; None when SubBlock is top-level
    page_idx         : int
    page_size        : Optional[list]
    bbox             : Optional[list]
    label            : str             # SubBlock role
    composite_label  : str             # parent Composite type
    reading_order    : Optional[int]   # block-level reading order
    score            : Optional[float]
    content          : str             # span text with inline $latex$ spliced in
    html             : Optional[str]   # table_body only
    image_path       : Optional[str]   # image/chart/seal/interline body only
    inline_equations : list            # inline formula LaTeX strings
    has_equation     : bool
    has_image        : bool
    level            : Optional[int]  = None  # heading level, titles only
    from_discarded   : bool = False
    sub_chunks       : Optional[list] = None


def build_section_tree(records: list[ChunkRecord]) -> dict[Optional[str], list[str]]:
    """Map each title's bbox_id to its direct children's bbox_ids in reading order.

    A title owns everything after it up to the next title of equal-or-higher level;
    a deeper title in between is itself a child and in turn owns its own children,
    so the map forms a recursive tree. The `None` key holds the roots: top-level
    titles plus any leading content before the first title. A title with no level
    can't be ranked, so it closes everything open and attaches as a root.

    The flat list stays the single source of truth -- this is a derived view over
    it (not stored on the records), so Document (section navigation) and Sampler
    (by-title draws) build on the same boundary rule without depending on each
    other.
    """
    parent_of: dict[str, Optional[str]] = {}
    stack: list[tuple[str, Optional[int]]] = []  # (title bbox_id, level) of open ancestors
    for r in records:
        if "title" in r.label:
            level = r.level
            while stack and (level is None or stack[-1][1] is None or stack[-1][1] >= level):
                stack.pop()
            parent_of[r.bbox_id] = stack[-1][0] if stack else None
            stack.append((r.bbox_id, level))
        else:
            parent_of[r.bbox_id] = stack[-1][0] if stack else None

    children: dict[Optional[str], list[str]] = defaultdict(list)
    for r in records:                       # second pass keeps children in reading order
        children[parent_of[r.bbox_id]].append(r.bbox_id)
    return dict(children)


def section_subtree(tree: dict[Optional[str], list[str]], root: str) -> list[str]:
    """A section root's bbox_id plus all its descendants in reading order -- preorder
    over the tree, whose children are already in reading order."""
    out = [root]
    stack = list(reversed(tree.get(root, [])))
    while stack:
        nid = stack.pop()
        out.append(nid)
        stack.extend(reversed(tree.get(nid, [])))
    return out


def iter_sections(records: list[ChunkRecord]) -> list[list[ChunkRecord]]:
    """Project the section tree into the document's top-level sections in reading
    order: the leading title-less run (if any) first, then each top-level title with
    its whole subtree folded in. Derived from build_section_tree so the boundary rule
    stays single-sourced; the sections are disjoint and cover every record, the way
    pages tile a document. This is the tree -> list view by-section sampling draws on.
    """
    tree = build_section_tree(records)
    by_id = {r.bbox_id: r for r in records}
    sections: list[list[ChunkRecord]] = []
    preamble: list[ChunkRecord] = []
    for rid in tree.get(None, []):
        if "title" in by_id[rid].label:
            sections.append([by_id[i] for i in section_subtree(tree, rid)])
        else:
            preamble.append(by_id[rid])   # leading run is contiguous at the front
    if preamble:
        sections.insert(0, preamble)
    return sections
