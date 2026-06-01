import json
import pathlib
from functools import cached_property
from typing import Optional


class Span:
    """Atomic content unit.

    type in {text, inline_equation, interline_equation, image, table, seal, chart}.
    Payload key varies by type:
      - text / inline_equation / interline_equation -> content (LaTeX lives here too)
      - image / chart -> image_path
      - table -> html (+ image_path)
    """

    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def type(self) -> str:
        return self.raw["type"]

    @property
    def bbox(self) -> Optional[list]:
        return self.raw.get("bbox")

    @property
    def content(self) -> Optional[str]:
        return self.raw.get("content")

    @property
    def html(self) -> Optional[str]:
        return self.raw.get("html")

    @property
    def image_path(self) -> Optional[str]:
        return self.raw.get("image_path")


class Line:
    """One visual line, owns its own bbox."""

    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def bbox(self) -> Optional[list]:
        return self.raw.get("bbox")

    @cached_property
    def spans(self) -> list[Span]:
        return [Span(s) for s in self.raw.get("spans", [])]


class SubBlock:
    """Semantic role unit.

    type in {text, title, table_body, table_caption, table_footnote,
             image_body, image_caption, image_footnote, ...}.
    """

    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def type(self) -> str:
        return self.raw["type"]

    @property
    def bbox(self) -> Optional[list]:
        return self.raw.get("bbox")

    @property
    def score(self) -> Optional[float]:
        return self.raw.get("score")

    @cached_property
    def lines(self) -> list[Line]:
        return [Line(line) for line in self.raw.get("lines", [])]


class CompositeBlock:
    """Top-level para_block.

    type in {text, title, table, image, chart, list, interline_equation, ...}.
    """

    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def type(self) -> str:
        return self.raw["type"]

    @property
    def bbox(self) -> Optional[list]:
        return self.raw.get("bbox")

    @property
    def score(self) -> Optional[float]:
        return self.raw.get("score")

    @property
    def index(self) -> Optional[int]:
        # Block-level reading order.
        return self.raw.get("index")

    @cached_property
    def sub_blocks(self) -> list[SubBlock]:
        # Composite types (table/image/chart) nest real SubBlocks under "blocks".
        # Simple types (text/title/...) carry "lines" directly; wrap the composite
        # itself as one implicit SubBlock so navigation is uniform across shapes.
        if "blocks" in self.raw:
            return [SubBlock(b) for b in self.raw["blocks"]]
        return [SubBlock(self.raw)]


class Page:
    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def page_idx(self) -> int:
        return self.raw["page_idx"]

    @property
    def page_size(self) -> Optional[list]:
        return self.raw.get("page_size")

    @cached_property
    def para_blocks(self) -> list[CompositeBlock]:
        # Default retrieval source: layout + caption/body paired, paragraph-merged.
        return [CompositeBlock(b) for b in self.raw.get("para_blocks", [])]


class MiddleJson:
    """Lazy navigable view over a MinerU <doc>_middle.json file.

    Layers: Page -> CompositeBlock -> SubBlock -> Line -> Span.
    """

    def __init__(self, raw: dict):
        self.raw = raw

    @classmethod
    def from_path(cls, path: pathlib.Path) -> "MiddleJson":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @cached_property
    def pages(self) -> list[Page]:
        return [Page(p) for p in self.raw.get("pdf_info", [])]
