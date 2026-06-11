import pathlib
from typing import Optional

from ..base import DocAnalyzer
from .chunk import ChunkRecord, flatten
from .middle import MiddleJson


# @dataclass
# class MinerUArtifact:
#     root: pathlib.Path
#     name : str

#     @classmethod
#     def from_dir(cls, path: pathlib.Path, name: Optional[str] = None) -> "MinerUArtifact":
#         return cls(root=path, name=name or path.parent.name)

#     @property
#     def middle_json(self) -> pathlib.Path:
#         return self.root / f"{self.name}_middle.json"

#     @property
#     def content_list_json(self) -> pathlib.Path:
#         return self.root / f"{self.name}_content_list.json"

#     @property
#     def images(self) -> list[pathlib.Path]:
#         return sorted((self.root / "images").glob("*.jpg"))

class MinerUArtifact(type(pathlib.Path())):

    @classmethod
    def from_dir(cls, path: pathlib.Path, basename: Optional[str] = None) -> "MinerUArtifact":
        self = cls(path)
        self._basename = basename
        return self

    @property
    def basename(self) -> str:
        # Explicit hint wins; else self-describe from the `*_middle.json` filename
        # (so a checkpoint's doc_id-named parent dir doesn't mislead us); else the
        # parent dir name, MinerU's conventional `<basename>/auto` layout.
        if getattr(self, "_basename", None):
            return self._basename
        middles = list(self.glob("*_middle.json"))
        if middles:
            return middles[0].name[: -len("_middle.json")]
        return self.parent.name

    @property
    def middle_json(self) -> pathlib.Path:
        return self / f"{self.basename}_middle.json"

    @property
    def model_json(self) -> pathlib.Path:
        return self / f"{self.basename}_model.json"

    @property
    def content_list_json(self) -> pathlib.Path:
        return self / f"{self.basename}_content_list.json"

    @property
    def layout_pdf(self) -> pathlib.Path:
        return self / f"{self.basename}_layout.pdf"

    @property
    def origin_pdf(self) -> pathlib.Path:
        return self / f"{self.basename}_origin.pdf"

    @property
    def span_pdf(self) -> pathlib.Path:
        return self / f"{self.basename}_span.pdf"

    @property
    def md(self) -> pathlib.Path:
        return self / f"{self.basename}.md"

    @property
    def images(self) -> list[pathlib.Path]:
        return sorted((self / "images").glob("*.jpg"))


class MinerUAnalyzer(DocAnalyzer):

    def __init__(self, source: pathlib.Path):
        # Accept the MinerU `auto` dir (or a MinerUArtifact); resolve the artifact once.
        self.source = source
        self.artifact = source if isinstance(source, MinerUArtifact) \
            else MinerUArtifact.from_dir(pathlib.Path(source))

    @property
    def default_doc_id(self) -> str:
        return self.artifact.basename

    def analyze(self, doc_id: Optional[str] = None) -> list[ChunkRecord]:
        middle = MiddleJson.from_path(self.artifact.middle_json)
        return flatten(middle, doc_id or self.default_doc_id)