"""End-to-end coverage for the FastAPI adapter via TestClient (no live server).

Drives every retrieval route in-process against `sample_auto_dir`, so this is
also the executable mirror of tests/manual/mcp_coverage.md: the adapters are thin
wrappers over the same Document, so green here means the tool surface works.
"""

import pytest
from fastapi.testclient import TestClient

from strata.app import create_app

DOC = "sample_doc"


@pytest.fixture
def client(sample_auto_dir, tmp_path):
    # A fresh in-process app per test, opening the fixture doc into a temp checkpoint.
    app = create_app(sources=[str(sample_auto_dir)], checkpoint_root=str(tmp_path / "ckpt"))
    return TestClient(app)


def test_health(client):
    assert client.get("/").json()["status"] == "ok"


def test_list_docs(client):
    assert {"doc_id": DOC, "n_pages": 1, "n_blocks": 4} in client.get("/docs/list").json()


def test_outline(client):
    entry = client.get(f"/docs/{DOC}/outline").json()[0]
    assert (entry["bbox_id"], entry["title"], entry["level"]) == ("p0_b0_s0_title", "Intro", 1)


def test_grep(client):
    hits = client.get(f"/docs/{DOC}/grep", params={"pattern": "holds"}).json()
    assert [h["bbox_id"] for h in hits] == ["p0_b1_s0_text"]


def test_grep_limit(client):
    assert len(client.get(f"/docs/{DOC}/grep", params={"pattern": ".", "limit": 1}).json()) == 1


def test_list_blocks(client):
    assert len(client.get(f"/docs/{DOC}/blocks", params={"page": 0}).json()) == 4
    titles = client.get(f"/docs/{DOC}/blocks", params={"label": "title"}).json()
    assert [b["bbox_id"] for b in titles] == ["p0_b0_s0_title"]


def test_read_block(client):
    assert client.get(f"/docs/{DOC}/blocks/p0_b1_s0_text").json()["content"] == "where $x_l$ holds"


def test_read_page(client):
    page = client.get(f"/docs/{DOC}/pages/0").json()
    assert page["page_idx"] == 0
    assert page["page_size"] == [600, 800]
    assert [r["kind"] for r in page["regions"]] == ["text", "text", "image"]  # caption folded away
    image = page["regions"][-1]
    assert image["caption"] == "Figure 1"
    assert image["content"] is None


def test_read_page_embed_images(client):
    image = client.get(f"/docs/{DOC}/pages/0", params={"embed_images": True}).json()["regions"][-1]
    assert image["content"].startswith("data:image/jpeg;base64,")


def test_page_info(client):
    info = client.get(f"/docs/{DOC}/pages/0/info").json()
    assert info["page_size"] == [600, 800]
    assert info["counts_by_label"]["title"] == 1


def test_context(client):
    ids = [b["bbox_id"] for b in client.get(
        f"/docs/{DOC}/blocks/p0_b1_s0_text/context", params={"n_prev": 1, "n_next": 1}
    ).json()]
    assert ids == ["p0_b0_s0_title", "p0_b1_s0_text", "p0_b2_s0_image_body"]


def test_parent(client):
    assert client.get(f"/docs/{DOC}/blocks/p0_b2_s0_image_body/parent").json() == "p0_b2"


def test_siblings(client):
    assert client.get(f"/docs/{DOC}/blocks/p0_b2_s0_image_body/siblings").json() == ["p0_b2_s1_image_caption"]


def test_next_prev(client):
    assert client.get(f"/docs/{DOC}/blocks/p0_b0_s0_title/next").json() == "p0_b1_s0_text"
    assert client.get(f"/docs/{DOC}/blocks/p0_b1_s0_text/prev").json() == "p0_b0_s0_title"


def test_open_and_close(client, sample_auto_dir):
    assert client.delete(f"/docs/{DOC}").status_code == 200
    assert client.get("/docs/list").json() == []
    reopened = client.post("/open", json={"source": str(sample_auto_dir)})
    assert reopened.json()["doc_id"] == DOC


def test_unknown_doc_is_404(client):
    assert client.get("/docs/nope/outline").status_code == 404


def test_unknown_block_is_404(client):
    assert client.get(f"/docs/{DOC}/blocks/p9_b9_s9_text").status_code == 404
