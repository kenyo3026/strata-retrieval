"""Filesystem persistence for opened documents.

A checkpoint store survives server restarts: each opened document is backed up
under a stable, doc_id-addressed directory, so re-pointing at the same root next
launch inherits everything already there. Pure addressing (Path subclasses, in
the strata / evals style); the copy / read is the store's job, domain lifecycle
stays in Main.

Layout:

    <root>/                 # Checkpoint -- never cleared, append-on-top
      <doc_id>/             # DocCheckpoint -- one opened document
        artifact/           # the provider artifact dir, copied verbatim
        meta.json           # how to re-open it (provider, source, opened_at)
"""

import json
import pathlib
import shutil
from datetime import datetime
from typing import Union


class DocCheckpoint(type(pathlib.Path())):
    """One opened document's on-disk backup: a uniform `artifact/` plus `meta.json`.

    The artifact is copied byte-for-byte (inner files keep their original names),
    so re-opening is just pointing the analyzer back at `artifact`. `meta.json`
    records the provider and provenance, not domain content.
    """

    @property
    def artifact(self) -> pathlib.Path:
        return self / "artifact"

    @property
    def meta_path(self) -> pathlib.Path:
        return self / "meta.json"

    def is_saved(self) -> bool:
        return self.meta_path.exists()

    def meta(self) -> dict:
        return json.loads(self.meta_path.read_text())

    def save(self, source: Union[str, pathlib.Path], doc_id: str, provider: str) -> "DocCheckpoint":
        # Back up the artifact dir verbatim and record how to re-open it.
        # Overwrites any existing checkpoint for this doc_id (re-backup on re-open).
        self.mkdir(parents=True, exist_ok=True)

        if self.artifact.exists():
            shutil.rmtree(self.artifact)

        shutil.copytree(str(source), self.artifact)

        self.meta_path.write_text(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "provider": provider,
                    "source": str(source),
                    "opened_at": datetime.now().isoformat(timespec="seconds"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return self


class Checkpoint(type(pathlib.Path())):
    """The persistent root: a flat collection of per-doc checkpoints.

    Made once by `new()` and never cleared. Pointing at an existing root inherits
    whatever doc checkpoints already live there.
    """

    @classmethod
    def new(cls, root: Union[str, pathlib.Path] = "checkpoint") -> "Checkpoint":
        self = cls(pathlib.Path(root))
        self.mkdir(parents=True, exist_ok=True)
        return self

    def doc(self, doc_id: str) -> DocCheckpoint:
        return DocCheckpoint(pathlib.Path(self) / doc_id)

    def doc_ids(self) -> list[str]:
        # Saved per-doc checkpoints in name order, for inherit-on-startup.
        return sorted(p.name for p in self.iterdir() if (p / "meta.json").exists())
