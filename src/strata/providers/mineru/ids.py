"""Deterministic bbox_id minting for MinerU blocks.

Stable across runs given the same middle.json: ids are derived purely from
positional indices in the para_blocks tree, so an agent can reuse a bbox_id
across turns and documents.

  composite : p{page_idx}_b{composite_index}
  subblock  : p{page_idx}_b{composite_index}_s{sub_index}_{role}

composite_index = position in page.para_blocks
sub_index       = position in composite.sub_blocks
"""


def composite_bbox_id(page_idx: int, composite_index: int) -> str:
    return f"p{page_idx}_b{composite_index}"


def subblock_bbox_id(page_idx: int, composite_index: int, sub_index: int, role: str) -> str:
    return f"{composite_bbox_id(page_idx, composite_index)}_s{sub_index}_{role}"
