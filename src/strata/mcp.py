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
    def grep(doc_id: str, pattern: str, ignore_case: bool = False) -> list:
        """Regex/substring search over block content (inline LaTeX included)."""
        return _dump(_doc(doc_id).grep(pattern, ignore_case=ignore_case))

    @mcp.tool
    def read_block(doc_id: str, bbox_id: str) -> dict:
        """Full content of one block by bbox_id."""
        try:
            return _dump(_doc(doc_id).read_block(bbox_id))
        except KeyError:
            raise ValueError(f"bbox_id '{bbox_id}' not found")

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
