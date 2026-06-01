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

    @app.get("/")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/open")
    def open_doc(request: OpenRequest):
        doc_id = main.open(request.source, doc_id=request.doc_id, provider=request.provider)
        return {"doc_id": doc_id}

    @app.delete("/docs/{doc_id}")
    def close_doc(doc_id: str):
        main.close(doc_id)
        return {"closed": doc_id}

    @app.get("/docs/{doc_id}/outline")
    def outline(doc_id: str):
        return _dump(_doc(doc_id).outline())

    @app.get("/docs/{doc_id}/grep")
    def grep(doc_id: str, pattern: str, ignore_case: bool = False):
        return _dump(_doc(doc_id).grep(pattern, ignore_case=ignore_case))

    @app.get("/docs/{doc_id}/blocks/{bbox_id}")
    def read_block(doc_id: str, bbox_id: str):
        try:
            return _dump(_doc(doc_id).read_block(bbox_id))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"bbox_id '{bbox_id}' not found")

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
