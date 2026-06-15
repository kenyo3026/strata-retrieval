# strata-retrieval MCP — tool coverage test script

Manual coverage script for the `strata-retrieval` MCP server. Paste the prompt
in section 1 into a Claude Code (or any MCP client) session that has the server
attached, or drive the tools directly with the args in section 2.

## Setup

Register the server (stdio) in the client's MCP config:

```json
{
  "mcpServers": {
    "strata-retrieval": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/Users/matthewk/workspace/new-things/doc-analysis/strata-retrieval",
        "strata-mcp"
      ]
    }
  }
}
```

On startup the server restores whatever is under `<project>/.strata/checkpoint`.
These cases assume `2512.24880v2` is already there, so `list_docs` shows it and
you can call tools with `doc_id = "2512.24880v2"` without opening first.

Reference ids used below: title `p0_b0_s0_title`, image `p0_b6_s0_image_body`
(parent `p0_b6`), pages `0..18`.

## 1. Natural-language script (drives every tool in order)

```
I have a strata-retrieval MCP attached. Test each tool in order and paste back
each result:

1. List the open documents (list_docs)
2. Re-open the document with source "checkpoint/2512.24880v2/artifact" (open)
3. Get the outline of doc_id "2512.24880v2" (outline)
4. Search this doc for "hyper", case-insensitive, at most 5 hits (grep)
5. List all blocks on page 0 (list_blocks, page=0)
6. List only blocks whose label is title (list_blocks, label="title")
7. Read the full content of block "p0_b0_s0_title" (read_block)
8. Get the page payload for page 0 (read_page, page_idx=0)
9. Get page 0 again with embed_images=true (large; just confirm the image
   region carries a data uri) (read_page)
10. Get page_info for page 0 — size / label counts / block ids (page_info)
11. Get the context around "p0_b0_s0_title" with n_prev=0, n_next=2 (context)
12. Get the parent of "p0_b6_s0_image_body" (parent → expect p0_b6)
13. Get the siblings of "p0_b6_s0_image_body" (siblings)
14. Get the next block after "p0_b0_s0_title" (next)
15. Get the previous block before "p0_b1_s0_text" (prev)
16. Close document "2512.24880v2" (close), then list_docs to confirm it's gone
```

## 2. Tool-by-tool arguments (14 tools)

| # | tool | args |
|---|---|---|
| 1 | `list_docs` | `{}` |
| 2 | `open` | `{"source":"checkpoint/2512.24880v2/artifact"}` |
| 3 | `outline` | `{"doc_id":"2512.24880v2"}` |
| 4 | `grep` | `{"doc_id":"2512.24880v2","pattern":"hyper","ignore_case":true,"limit":5}` |
| 5 | `list_blocks` | `{"doc_id":"2512.24880v2","page":0}` |
| 6 | `read_block` | `{"doc_id":"2512.24880v2","bbox_id":"p0_b0_s0_title"}` |
| 7 | `read_page` | `{"doc_id":"2512.24880v2","page_idx":0}` |
| 8 | `read_page` (embed) | `{"doc_id":"2512.24880v2","page_idx":0,"embed_images":true}` |
| 9 | `page_info` | `{"doc_id":"2512.24880v2","page_idx":0}` |
| 10 | `context` | `{"doc_id":"2512.24880v2","bbox_id":"p0_b0_s0_title","n_prev":0,"n_next":2}` |
| 11 | `parent` | `{"doc_id":"2512.24880v2","bbox_id":"p0_b6_s0_image_body"}` → `"p0_b6"` |
| 12 | `siblings` | `{"doc_id":"2512.24880v2","bbox_id":"p0_b6_s0_image_body"}` |
| 13 | `next` | `{"doc_id":"2512.24880v2","bbox_id":"p0_b0_s0_title"}` → `"p0_b1_s0_text"` |
| 14 | `prev` | `{"doc_id":"2512.24880v2","bbox_id":"p0_b1_s0_text"}` → `"p0_b0_s0_title"` |
| 15 | `close` | `{"doc_id":"2512.24880v2"}` |

## 3. Error paths (graceful-failure coverage)

| case | args | expected |
|---|---|---|
| unknown doc_id | `outline {"doc_id":"nope"}` | `ValueError: doc_id 'nope' is not open` |
| unknown bbox_id | `read_block {"doc_id":"2512.24880v2","bbox_id":"p9_b99_s0_text"}` | `ValueError: bbox_id '...' not found` |
| top-level parent | `parent {"doc_id":"2512.24880v2","bbox_id":"p0_b0_s0_title"}` | `null` (not an error) |

## Notes

- **#8 embed_images=true is large** (the whole image as base64). Confirm the
  image region's `content` starts with `data:image/jpeg;base64,`; no need to
  paste it all.
- If you launch with the absolute venv script instead of `uv run --directory`,
  the process cwd differs: the relative `open` source path and the checkpoint
  location both change. Prefer the `uv run --directory` form above.
