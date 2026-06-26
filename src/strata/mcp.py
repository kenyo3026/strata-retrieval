"""FastMCP adapter for strata-retrieval.

Thin wrapper over a single stateful Main: opened documents are cached for the
life of the server and addressed by doc_id. Holds no domain logic.

Every tool builds and returns its own `ToolResult` explicitly -- a JSON text block,
plus structuredContent for object payloads -- rather than leaning on FastMCP's
implicit value->ContentBlock conversion, so the wire shape each tool produces is
spelled out in place. The one tool that goes beyond a JSON text block is
`read_page` with embed_images, which also emits ImageContent.
"""

import argparse
import json
from dataclasses import asdict, is_dataclass
from typing import Optional

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from mcp.types import ImageContent, TextContent

from .main import Main
from .providers.factory import ProviderType
from .providers.document import RegionKind


def _dump(result):
    # Dataclass results -> plain dict / list (the JSON-able payload).
    if isinstance(result, list):
        return [asdict(item) if is_dataclass(item) else item for item in result]
    return asdict(result) if is_dataclass(result) else result


def _data_uri_parts(uri: str):
    # Split a "data:<mime>;base64,<payload>" uri back into (mime, base64 payload).
    header, _, payload = uri.partition(",")
    mime = header[len("data:"):].split(";")[0] or "image/jpeg"
    return mime, payload


def create_mcp_server() -> FastMCP:
    mcp = FastMCP("strata-retrieval")
    main = Main()

    def _doc(doc_id: str):
        try:
            return main.doc(doc_id)
        except KeyError:
            raise ToolError(f"doc_id '{doc_id}' is not open")

    def _resolve(bbox_id: str, thunk):
        # Run a block-keyed lookup, turning an unknown bbox_id into a clear error.
        try:
            return _dump(thunk())
        except KeyError:
            raise ToolError(f"bbox_id '{bbox_id}' not found")

    @mcp.tool(name="open")
    def open_doc(source: str, doc_id: Optional[str] = None, provider: str = ProviderType.MINERU) -> ToolResult:
        """Parse a document `auto` dir into a flat index; returns its doc_id."""
        payload = {"doc_id": main.open(source, doc_id=doc_id, provider=provider)}
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
            structured_content=payload,
        )

    @mcp.tool
    def close(doc_id: str) -> ToolResult:
        """Release a cached document."""
        main.close(doc_id)
        payload = {"closed": doc_id}
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
            structured_content=payload,
        )

    @mcp.tool
    def outline(doc_id: str) -> ToolResult:
        """Title hierarchy (TOC) in document order."""
        payload = _dump(_doc(doc_id).outline())
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
            structured_content=None,
        )

    @mcp.tool
    def grep(doc_id: str, pattern: str, ignore_case: bool = False, limit: Optional[int] = None) -> ToolResult:
        """Regex/substring search over block content (inline LaTeX included)."""
        payload = _dump(_doc(doc_id).grep(pattern, ignore_case=ignore_case, limit=limit))
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
            structured_content=None,
        )

    @mcp.tool(name="list_docs")
    def list_docs() -> ToolResult:
        """Lightweight overview (id + counts) of every open document."""
        payload = _dump(main.doc_summaries())
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
            structured_content=None,
        )

    @mcp.tool
    def list_blocks(doc_id: str, label: Optional[str] = None, page: Optional[int] = None) -> ToolResult:
        """Compact block listing, optionally filtered by label and/or page index."""
        payload = _dump(_doc(doc_id).list_blocks(label=label, page=page))
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
            structured_content=None,
        )

    @mcp.tool
    def read_block(doc_id: str, bbox_id: str) -> ToolResult:
        """Full content of one block by bbox_id."""
        payload = _resolve(bbox_id, lambda: _doc(doc_id).read_block(bbox_id))
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
            structured_content=payload,
        )

    @mcp.tool
    def read_page(doc_id: str, page_idx: int, embed_images: bool = False) -> ToolResult:
        """Whole page as ordered per-kind regions.

        Default: the structured payload (image regions are placeholders). With
        embed_images: the same payload as structured_content, plus reading-order
        content blocks where each image is a real ImageContent the model can see.
        """
        doc = _doc(doc_id)
        if not embed_images:
            payload = _dump(doc.read_page(page_idx))
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
                structured_content=payload,
            )

        placeholders = doc.read_page(page_idx) # structured_content, no bytes
        blocks = []
        for region in doc.read_page(page_idx, embed_images=True).regions:
            if region.kind == RegionKind.IMAGE:
                caption = f": {region.caption}" if region.caption else ""
                blocks.append(TextContent(type="text", text=f"[image {region.bbox_id}]{caption}"))
                if region.content:
                    mime, data = _data_uri_parts(region.content)
                    # ImageContent (not EmbeddedResource): widest multimodal-client support, and the bbox_id binding already lives in the adjacent text + structured_content.
                    blocks.append(ImageContent(type="image", data=data, mimeType=mime))
            elif region.content:
                blocks.append(TextContent(type="text", text=region.content))
        return ToolResult(content=blocks, structured_content=_dump(placeholders))

    @mcp.tool
    def page_info(doc_id: str, page_idx: int) -> ToolResult:
        """Page metadata: size, per-label counts, and block ids."""
        payload = _dump(_doc(doc_id).page_info(page_idx))
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
            structured_content=payload,
        )

    @mcp.tool
    def read_section(doc_id: str, bbox_id: str, embed_images: bool = False) -> ToolResult:
        """A title and its whole subtree as ordered per-kind regions -- the section
        peer of read_page, keyed by a title block.

        Default: the structured payload (image regions are placeholders). With
        embed_images: the same payload as structured_content, plus reading-order
        content blocks where each image is a real ImageContent the model can see.
        """
        doc = _doc(doc_id)
        if not embed_images:
            payload = _resolve(bbox_id, lambda: doc.read_section(bbox_id))
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
                structured_content=payload,
            )

        placeholders = _resolve(bbox_id, lambda: doc.read_section(bbox_id))  # also validates bbox_id
        blocks = []
        for region in doc.read_section(bbox_id, embed_images=True).regions:
            if region.kind == RegionKind.IMAGE:
                caption = f": {region.caption}" if region.caption else ""
                blocks.append(TextContent(type="text", text=f"[image {region.bbox_id}]{caption}"))
                if region.content:
                    mime, data = _data_uri_parts(region.content)
                    blocks.append(ImageContent(type="image", data=data, mimeType=mime))
            elif region.content:
                blocks.append(TextContent(type="text", text=region.content))
        return ToolResult(content=blocks, structured_content=placeholders)

    @mcp.tool
    def context(doc_id: str, bbox_id: str, n_prev: int = 1, n_next: int = 1) -> ToolResult:
        """A block plus its n_prev/n_next reading-order neighbours."""
        payload = _resolve(bbox_id, lambda: _doc(doc_id).read_block_with_context(bbox_id, n_prev=n_prev, n_next=n_next))
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
            structured_content=None,
        )

    @mcp.tool
    def parent(doc_id: str, bbox_id: str) -> ToolResult:
        """Composite parent id of a block, or null when it is top-level."""
        payload = _resolve(bbox_id, lambda: _doc(doc_id).parent(bbox_id))
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
            structured_content=None,
        )

    @mcp.tool
    def siblings(doc_id: str, bbox_id: str) -> ToolResult:
        """Co-members under the same composite block."""
        payload = _resolve(bbox_id, lambda: _doc(doc_id).siblings(bbox_id))
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))],
            structured_content=None,
        )

    @mcp.tool(name="next")
    def next_block(doc_id: str, bbox_id: str) -> ToolResult:
        """The next block in reading order, or null at the end."""
        payload = _resolve(bbox_id, lambda: _doc(doc_id).next(bbox_id))
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
            structured_content=None,
        )

    @mcp.tool(name="prev")
    def prev_block(doc_id: str, bbox_id: str) -> ToolResult:
        """The previous block in reading order, or null at the start."""
        payload = _resolve(bbox_id, lambda: _doc(doc_id).prev(bbox_id))
        return ToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
            structured_content=None,
        )

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
