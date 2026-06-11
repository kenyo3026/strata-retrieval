"""FastAPI adapter for strata-retrieval.

Thin wrapper over a single stateful Main: opened documents are cached in memory
for the life of the server and addressed by doc_id. Holds no domain logic.

pydantic is used only where FastAPI requires it (the POST /open request body);
responses are plain dataclass-derived dicts, not pydantic models.
"""

import argparse
from dataclasses import asdict, is_dataclass
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .main import Main
from .providers.factory import ProviderType


class OpenRequest(BaseModel):
    source: str
    doc_id: Optional[str] = None
    provider: str = ProviderType.MINERU


def _dump(result):
    if isinstance(result, list):
        return [asdict(item) if is_dataclass(item) else item for item in result]
    return asdict(result) if is_dataclass(result) else result


def create_app() -> FastAPI:
    app = FastAPI(title="strata-retrieval", version="0.1.0")
    main = Main()

    def _doc(doc_id: str):
        try:
            return main.doc(doc_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"doc_id '{doc_id}' is not open")

    def _resolve(bbox_id: str, thunk):
        # Run a block-keyed lookup, mapping an unknown bbox_id to a 404.
        try:
            return _dump(thunk())
        except KeyError:
            raise HTTPException(status_code=404, detail=f"bbox_id '{bbox_id}' not found")

    @app.get("/")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/open")
    def open_doc(request: OpenRequest):
        doc_id = main.open(request.source, doc_id=request.doc_id, provider=request.provider)
        return {"doc_id": doc_id}

    @app.get("/docs/list")
    def list_docs():
        return _dump(main.doc_summaries())

    @app.delete("/docs/{doc_id}")
    def close_doc(doc_id: str):
        main.close(doc_id)
        return {"closed": doc_id}

    @app.get("/docs/{doc_id}/outline")
    def outline(doc_id: str):
        return _dump(_doc(doc_id).outline())

    @app.get("/docs/{doc_id}/grep")
    def grep(doc_id: str, pattern: str, ignore_case: bool = False, limit: Optional[int] = None):
        return _dump(_doc(doc_id).grep(pattern, ignore_case=ignore_case, limit=limit))

    @app.get("/docs/{doc_id}/blocks")
    def list_blocks(doc_id: str, label: Optional[str] = None, page: Optional[int] = None):
        return _dump(_doc(doc_id).list_blocks(label=label, page=page))

    @app.get("/docs/{doc_id}/blocks/{bbox_id}")
    def read_block(doc_id: str, bbox_id: str):
        return _resolve(bbox_id, lambda: _doc(doc_id).read_block(bbox_id))

    @app.get("/docs/{doc_id}/blocks/{bbox_id}/context")
    def read_block_with_context(doc_id: str, bbox_id: str, n_prev: int = 1, n_next: int = 1):
        return _resolve(bbox_id, lambda: _doc(doc_id).read_block_with_context(bbox_id, n_prev=n_prev, n_next=n_next))

    @app.get("/docs/{doc_id}/blocks/{bbox_id}/parent")
    def parent(doc_id: str, bbox_id: str):
        return _resolve(bbox_id, lambda: _doc(doc_id).parent(bbox_id))

    @app.get("/docs/{doc_id}/blocks/{bbox_id}/siblings")
    def siblings(doc_id: str, bbox_id: str):
        return _resolve(bbox_id, lambda: _doc(doc_id).siblings(bbox_id))

    @app.get("/docs/{doc_id}/blocks/{bbox_id}/next")
    def next_block(doc_id: str, bbox_id: str):
        return _resolve(bbox_id, lambda: _doc(doc_id).next(bbox_id))

    @app.get("/docs/{doc_id}/blocks/{bbox_id}/prev")
    def prev_block(doc_id: str, bbox_id: str):
        return _resolve(bbox_id, lambda: _doc(doc_id).prev(bbox_id))

    @app.get("/docs/{doc_id}/pages/{page_idx}")
    def read_page(doc_id: str, page_idx: int):
        return _dump(_doc(doc_id).read_page(page_idx))

    @app.get("/docs/{doc_id}/pages/{page_idx}/info")
    def page_info(doc_id: str, page_idx: int):
        return _dump(_doc(doc_id).page_info(page_idx))

    return app


app = create_app()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the strata-retrieval API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
