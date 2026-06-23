"""The provider-agnostic record contract.

`ChunkRecord` is the normalization boundary: every provider's analyzer flattens
its own raw format into this shape, and every consumer (Document, store, future
sampler) reads only this -- nothing downstream touches a provider's raw structure.
The standard vocabulary (RegionKind / role) that lets consumers stop sniffing a
provider's open label strings will land here too.
"""

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
