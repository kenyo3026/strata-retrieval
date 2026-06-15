"""Pytest fixtures for strata-retrieval tests.

`sample_auto_dir` builds a minimal but representative MinerU `auto` directory in
a temp location, so tests are self-contained (no dependency on the examples corpus).
It covers the key flatten paths: a title with a level, a text line mixing text and
an inline equation, and an image composite (body + caption).
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

_MIDDLE = {
    "pdf_info": [
        {
            "page_idx": 0,
            "page_size": [600, 800],
            "preproc_blocks": [],
            "discarded_blocks": [],
            "para_blocks": [
                {
                    "type": "title", "bbox": [0, 0, 100, 20], "index": 1, "score": 0.9, "level": 1,
                    "lines": [{"bbox": [0, 0, 100, 20], "spans": [
                        {"type": "text", "bbox": [0, 0, 100, 20], "content": "Intro"}
                    ]}],
                },
                {
                    "type": "text", "bbox": [0, 30, 200, 60], "index": 2, "score": 0.95,
                    "lines": [{"bbox": [0, 30, 200, 45], "spans": [
                        {"type": "text", "bbox": [0, 30, 40, 45], "content": "where"},
                        {"type": "inline_equation", "bbox": [40, 30, 60, 45], "content": "x_l"},
                        {"type": "text", "bbox": [60, 30, 200, 45], "content": "holds"},
                    ]}],
                },
                {
                    "type": "image", "bbox": [0, 70, 300, 300], "index": 3, "score": 0.8,
                    "blocks": [
                        {"type": "image_body", "bbox": [0, 70, 300, 250], "index": 3, "score": 0.8,
                         "lines": [{"bbox": [0, 70, 300, 250], "spans": [
                             {"type": "image", "bbox": [0, 70, 300, 250], "image_path": "img0.jpg"}
                         ]}]},
                        {"type": "image_caption", "bbox": [0, 255, 300, 270], "index": 3, "score": 0.7,
                         "lines": [{"bbox": [0, 255, 300, 270], "spans": [
                             {"type": "text", "bbox": [0, 255, 300, 270], "content": "Figure 1"}
                         ]}]},
                    ],
                },
            ],
        }
    ],
    "_backend": "pipeline",
    "_version_name": "test",
}


@pytest.fixture
def sample_auto_dir():
    tmp = Path(tempfile.mkdtemp())
    doc_name = "sample_doc"
    auto = tmp / doc_name / "auto"
    auto.mkdir(parents=True)
    (auto / f"{doc_name}_middle.json").write_text(json.dumps(_MIDDLE), encoding="utf-8")
    images = auto / "images"
    images.mkdir()
    (images / "img0.jpg").write_bytes(b"\xff\xd8\xff\xe0stub-jpeg")  # for embed_images
    yield auto
    shutil.rmtree(tmp, ignore_errors=True)
