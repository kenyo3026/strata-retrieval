"""Flatten a MinerU middle.json into a flat list of ChunkRecord.

One record per SubBlock, with its Composite block as parent grouping. See
.prps/mineru_rag.md (section 5) and .prps/mineru_middle_schema.md.

Scope (stage 1 / step 4): para_blocks only. sub_chunks (large-block splitting,
PRP 5.1) and content_md are deferred until a real need appears.
"""

import re
from dataclasses import dataclass
from typing import Optional

from .ids import composite_bbox_id, subblock_bbox_id
from .middle import MiddleJson, SubBlock

# Span types whose payload is text destined for `content`.
TEXT_SPAN = "text"
INLINE_EQUATION_SPAN = "inline_equation"
INTERLINE_EQUATION_SPAN = "interline_equation"

# Span types whose payload is a cropped-image reference.
TABLE_SPAN = "table"
IMAGE_SPANS = ("image", "chart", "seal")

# CJK chars (Hiragana, Katakana, CJK ideographs incl. ext-A, compat, Hangul).
# A block containing any of these joins its span fragments tight; otherwise the
# fragments join with a single space.
_CJK_RE = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


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
    from_discarded   : bool = False
    sub_chunks       : Optional[list] = None


def _extract_payload(sub_block: SubBlock):
    """Walk a SubBlock's spans into (content, html, image_path, inline_equations).

    Body SubBlocks (image/table/chart) yield content == "" -- their payload is the
    cropped image / html. image_path and html are take-first (PRP decision A:
    MinerU itself is one-body-one-image).
    """
    parts = []
    text_for_lang = []
    inline_equations = []
    html = None
    image_path = None
    has_interline = False

    for line in sub_block.lines:

        for span in line.spans:
            span_type = span.type

            if span_type == TEXT_SPAN:
                text = (span.content or "").strip()
                if text:
                    parts.append(text)
                    text_for_lang.append(text)

            elif span_type == INLINE_EQUATION_SPAN:
                latex = (span.content or "").strip()
                if latex:
                    parts.append(f"${latex}$")
                    inline_equations.append(latex)

            elif span_type == INTERLINE_EQUATION_SPAN:
                has_interline = True
                latex = (span.content or "").strip()

                if latex:
                    parts.append(f"$${latex}$$")

                if image_path is None and span.image_path:
                    image_path = span.image_path

            elif span_type == TABLE_SPAN:

                if html is None and span.html:
                    html = span.html

                if image_path is None and span.image_path:
                    image_path = span.image_path

            elif span_type in IMAGE_SPANS:
                if image_path is None and span.image_path:
                    image_path = span.image_path

    # Spacing only needs the CJK-vs-latin split: CJK joins tight, latin with a
    # single space (which also separates inline-equation seams cleanly).
    # TODO: if the project later adopts fast_langdetect for other reasons, align
    # this seam logic with MinerU's merge_para_with_text (full lang-aware spacing
    # + latin line-end hyphen merge). Deferred -- pulling a fasttext model
    # (0.9-126MB + native lib) only for spacing is overhead. See
    # .prps/mineru_middle_schema.md.
    separator = "" if _CJK_RE.search("".join(text_for_lang)) else " "
    content = separator.join(parts).strip()
    has_equation = bool(inline_equations) or has_interline
    return content, html, image_path, inline_equations, has_equation


def flatten(middle: MiddleJson, doc_id: str) -> list[ChunkRecord]:
    records = []

    for page in middle.pages:

        for composite_index, composite in enumerate(page.para_blocks):
            composite_id = composite_bbox_id(page.page_idx, composite_index)
            is_nested = "blocks" in composite.raw

            for sub_index, sub_block in enumerate(composite.sub_blocks):
                content, html, image_path, inline_equations, has_equation = _extract_payload(sub_block)
                records.append(
                    ChunkRecord(
                        doc_id=doc_id,
                        bbox_id=subblock_bbox_id(page.page_idx, composite_index, sub_index, sub_block.type),
                        parent_bbox_id=composite_id if is_nested else None,
                        page_idx=page.page_idx,
                        page_size=page.page_size,
                        bbox=sub_block.bbox,
                        label=sub_block.type,
                        composite_label=composite.type,
                        reading_order=composite.index,
                        score=sub_block.score,
                        content=content,
                        html=html,
                        image_path=image_path,
                        inline_equations=inline_equations,
                        has_equation=has_equation,
                        has_image=image_path is not None,
                    )
                )

    return records
