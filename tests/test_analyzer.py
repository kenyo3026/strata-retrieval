"""Unit tests for MinerUAnalyzer (ingestion: middle.json -> ChunkRecord[]).

Tests cover:
- analyze() returns ChunkRecord[]
- doc_id resolution: default (basename) vs explicit override
- accepts both an `auto` dir path and a MinerUArtifact
- flatten correctness: title level, inline-equation splicing, image-body payload,
  deterministic bbox_ids and parent grouping

Usage:
    pytest tests/test_analyzer.py -v
"""

import sys
from pathlib import Path

# Resolve `strata` whether run via pytest or as a standalone module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strata.providers.mineru.analyzer import MinerUAnalyzer, MinerUArtifact
from strata.providers.record import ChunkRecord


class TestMinerUAnalyzer:
    """Ingestion behaviour of MinerUAnalyzer."""

    def test_analyze_returns_chunk_records(self, sample_auto_dir):
        records = MinerUAnalyzer(sample_auto_dir).analyze()
        assert records
        assert all(isinstance(r, ChunkRecord) for r in records)

    def test_default_doc_id_is_basename(self, sample_auto_dir):
        assert MinerUAnalyzer(sample_auto_dir).default_doc_id == "sample_doc"

    def test_doc_id_defaults_to_basename(self, sample_auto_dir):
        records = MinerUAnalyzer(sample_auto_dir).analyze()
        assert all(r.doc_id == "sample_doc" for r in records)

    def test_doc_id_override(self, sample_auto_dir):
        records = MinerUAnalyzer(sample_auto_dir).analyze(doc_id="custom")
        assert all(r.doc_id == "custom" for r in records)

    def test_accepts_artifact_or_path(self, sample_auto_dir):
        from_path = MinerUAnalyzer(sample_auto_dir).analyze()
        from_artifact = MinerUAnalyzer(MinerUArtifact.from_dir(sample_auto_dir)).analyze()
        assert [r.bbox_id for r in from_path] == [r.bbox_id for r in from_artifact]

    def test_title_level_captured(self, sample_auto_dir):
        records = MinerUAnalyzer(sample_auto_dir).analyze()
        title = next(r for r in records if r.label == "title")
        assert title.level == 1
        assert title.content == "Intro"

    def test_inline_equation_spliced_into_content(self, sample_auto_dir):
        records = MinerUAnalyzer(sample_auto_dir).analyze()
        text = next(r for r in records if r.label == "text")
        assert "$x_l$" in text.content
        assert text.inline_equations == ["x_l"]
        assert text.has_equation is True

    def test_image_body_payload(self, sample_auto_dir):
        records = MinerUAnalyzer(sample_auto_dir).analyze()
        body = next(r for r in records if r.label == "image_body")
        assert body.content == ""
        assert body.image_path == "img0.jpg"
        assert body.has_image is True
        assert body.parent_bbox_id == "p0_b2"

    def test_deterministic_bbox_ids(self, sample_auto_dir):
        ids = [r.bbox_id for r in MinerUAnalyzer(sample_auto_dir).analyze()]
        assert ids == [
            "p0_b0_s0_title",
            "p0_b1_s0_text",
            "p0_b2_s0_image_body",
            "p0_b2_s1_image_caption",
        ]


if __name__ == "__main__":
    from .base import run_tests_with_report
    sys.exit(run_tests_with_report(__file__, "analyzer"))
