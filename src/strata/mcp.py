"""FastMCP adapter for strata-retrieval.

Thin wrapper over a single stateful Main: opened documents are cached for the
life of the server and addressed by doc_id. Holds no domain logic. Tool results
are plain dataclass-derived dicts/lists; FastMCP handles the wire format.
"""

import argparse
from dataclasses import asdict, is_dataclass
from typing import Optional

from fastmcp import FastMCP

from .main import Main
from .providers.factory import ProviderType


def _dump(result):
    if isinstance(result, list):
        return [asdict(item) if is_dataclass(item) else item for item in result]
    return asdict(result) if is_dataclass(result) else result


def create_mcp_server() -> FastMCP:
    mcp = FastMCP("strata-retrieval")
    main = Main()

    def _doc(doc_id: str):
        try:
            return main.doc(doc_id)
        except KeyError:
            raise ValueError(f"doc_id '{doc_id}' is not open")

    def _resolve(bbox_id: str, thunk):
        # Run a block-keyed lookup, turning an unknown bbox_id into a clear error.
        try:
            return _dump(thunk())
        except KeyError:
            raise ValueError(f"bbox_id '{bbox_id}' not found")

    @mcp.tool(name="open")
    def open_doc(source: str, doc_id: Optional[str] = None, provider: str = ProviderType.MINERU) -> dict:
        """Parse a document `auto` dir into a flat index; returns its doc_id."""
        return {"doc_id": main.open(source, doc_id=doc_id, provider=provider)}

    @mcp.tool
    def close(doc_id: str) -> dict:
        """Release a cached document."""
        main.close(doc_id)
        return {"closed": doc_id}

    @mcp.tool
    def outline(doc_id: str) -> list:
        """Title hierarchy (TOC) in document order."""
        return _dump(_doc(doc_id).outline())

    @mcp.tool
    def grep(doc_id: str, pattern: str, ignore_case: bool = False, limit: Optional[int] = None) -> list:
        """Regex/substring search over block content (inline LaTeX included)."""
        return _dump(_doc(doc_id).grep(pattern, ignore_case=ignore_case, limit=limit))

    @mcp.tool(name="list_docs")
    def list_docs() -> list:
        """Lightweight overview (id + counts) of every open document."""
        return _dump(main.doc_summaries())

    @mcp.tool
    def list_blocks(doc_id: str, label: Optional[str] = None, page: Optional[int] = None) -> list:
        """Compact block listing, optionally filtered by label and/or page index."""
        return _dump(_doc(doc_id).list_blocks(label=label, page=page))

    @mcp.tool
    def read_block(doc_id: str, bbox_id: str) -> dict:
        """Full content of one block by bbox_id."""
        return _resolve(bbox_id, lambda: _doc(doc_id).read_block(bbox_id))

    @mcp.tool
    def read_page(doc_id: str, page_idx: int, embed_images: bool = False) -> dict:
        """Whole page as ordered per-kind regions; embed_images inlines image bytes as data uris."""
        return _dump(_doc(doc_id).read_page(page_idx, embed_images=embed_images))

    @mcp.tool
    def page_info(doc_id: str, page_idx: int) -> dict:
        """Page metadata: size, per-label counts, and block ids."""
        return _dump(_doc(doc_id).page_info(page_idx))

    @mcp.tool
    def context(doc_id: str, bbox_id: str, n_prev: int = 1, n_next: int = 1) -> list:
        """A block plus its n_prev/n_next reading-order neighbours."""
        return _resolve(bbox_id, lambda: _doc(doc_id).read_block_with_context(bbox_id, n_prev=n_prev, n_next=n_next))

    @mcp.tool
    def parent(doc_id: str, bbox_id: str) -> Optional[str]:
        """Composite parent id of a block, or null when it is top-level."""
        return _resolve(bbox_id, lambda: _doc(doc_id).parent(bbox_id))

    @mcp.tool
    def siblings(doc_id: str, bbox_id: str) -> list:
        """Co-members under the same composite block."""
        return _resolve(bbox_id, lambda: _doc(doc_id).siblings(bbox_id))

    @mcp.tool(name="next")
    def next_block(doc_id: str, bbox_id: str) -> Optional[str]:
        """The next block in reading order, or null at the end."""
        return _resolve(bbox_id, lambda: _doc(doc_id).next(bbox_id))

    @mcp.tool(name="prev")
    def prev_block(doc_id: str, bbox_id: str) -> Optional[str]:
        """The previous block in reading order, or null at the start."""
        return _resolve(bbox_id, lambda: _doc(doc_id).prev(bbox_id))

    return mcp


mcp = create_mcp_server()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the strata-retrieval MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
