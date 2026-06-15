"""CLI adapter for strata-retrieval.

Thin wrapper over Main: each command opens a document, runs one retrieval tool,
and prints the result as JSON (one object per line for lists). Holds no domain
logic. Serialization (dataclass -> dict -> JSON) lives here, at the adapter edge.
"""

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass

from .main import Main


def _open(source: str):
    main = Main()
    return main.doc(main.open(source))


def _to_jsonable(obj):
    return asdict(obj) if is_dataclass(obj) else obj


def _dump(result) -> None:
    if isinstance(result, list):
        for item in result:
            print(json.dumps(_to_jsonable(item), ensure_ascii=False))
    else:
        print(json.dumps(_to_jsonable(result), ensure_ascii=False))


def cmd_outline(args) -> int:
    _dump(_open(args.source).outline())
    return 0


def cmd_grep(args) -> int:
    _dump(_open(args.source).grep(args.pattern, ignore_case=args.ignore_case, limit=args.limit))
    return 0


def cmd_read_block(args) -> int:
    _dump(_open(args.source).read_block(args.bbox_id))
    return 0


def cmd_read_page(args) -> int:
    _dump(_open(args.source).read_page(args.page_idx, embed_images=args.embed_images))
    return 0


def cmd_list_docs(args) -> int:
    main = Main()
    main.open(args.source)
    _dump(main.doc_summaries())
    return 0


def cmd_list_blocks(args) -> int:
    _dump(_open(args.source).list_blocks(label=args.label, page=args.page))
    return 0


def cmd_page_info(args) -> int:
    _dump(_open(args.source).page_info(args.page_idx))
    return 0


def cmd_context(args) -> int:
    _dump(_open(args.source).read_block_with_context(args.bbox_id, n_prev=args.n_prev, n_next=args.n_next))
    return 0


def cmd_parent(args) -> int:
    _dump(_open(args.source).parent(args.bbox_id))
    return 0


def cmd_siblings(args) -> int:
    _dump(_open(args.source).siblings(args.bbox_id))
    return 0


def cmd_next(args) -> int:
    _dump(_open(args.source).next(args.bbox_id))
    return 0


def cmd_prev(args) -> int:
    _dump(_open(args.source).prev(args.bbox_id))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="strata", description="MinerU retrieval tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = argparse.ArgumentParser(add_help=False)
    source.add_argument("-s", "--source", required=True, help="MinerU `auto` directory")

    block = argparse.ArgumentParser(add_help=False)
    block.add_argument("bbox_id", help="Stable block id, e.g. p0_b6_s0_image_body")

    p_outline = subparsers.add_parser("outline", parents=[source], help="Title hierarchy (TOC)")
    p_outline.set_defaults(func=cmd_outline)

    p_grep = subparsers.add_parser("grep", parents=[source], help="Regex search over block content")
    p_grep.add_argument("pattern", help="Substring or regex pattern")
    p_grep.add_argument("-i", "--ignore-case", action="store_true", help="Case-insensitive match")
    p_grep.add_argument("--limit", type=int, help="Cap the number of matched blocks")
    p_grep.set_defaults(func=cmd_grep)

    p_docs = subparsers.add_parser("list-docs", parents=[source], help="Overview of open documents")
    p_docs.set_defaults(func=cmd_list_docs)

    p_blocks = subparsers.add_parser("list-blocks", parents=[source], help="Compact block listing")
    p_blocks.add_argument("--label", help="Filter by label")
    p_blocks.add_argument("--page", type=int, help="Filter by page index")
    p_blocks.set_defaults(func=cmd_list_blocks)

    p_read = subparsers.add_parser("read-block", parents=[source, block], help="Full content of one block")
    p_read.set_defaults(func=cmd_read_block)

    p_page = subparsers.add_parser("read-page", parents=[source], help="Whole page as ordered regions")
    p_page.add_argument("page_idx", type=int, help="0-based page index")
    p_page.add_argument("--embed-images", action="store_true", help="Inline image bytes as base64 data uris")
    p_page.set_defaults(func=cmd_read_page)

    p_pinfo = subparsers.add_parser("page-info", parents=[source], help="Page size, label counts, block ids")
    p_pinfo.add_argument("page_idx", type=int, help="0-based page index")
    p_pinfo.set_defaults(func=cmd_page_info)

    p_ctx = subparsers.add_parser("context", parents=[source, block], help="A block plus its neighbours")
    p_ctx.add_argument("--n-prev", type=int, default=1, help="Preceding blocks to include")
    p_ctx.add_argument("--n-next", type=int, default=1, help="Following blocks to include")
    p_ctx.set_defaults(func=cmd_context)

    p_parent = subparsers.add_parser("parent", parents=[source, block], help="Composite parent id, or null")
    p_parent.set_defaults(func=cmd_parent)

    p_sib = subparsers.add_parser("siblings", parents=[source, block], help="Co-members under the same composite")
    p_sib.set_defaults(func=cmd_siblings)

    p_next = subparsers.add_parser("next", parents=[source, block], help="Next block in reading order, or null")
    p_next.set_defaults(func=cmd_next)

    p_prev = subparsers.add_parser("prev", parents=[source, block], help="Previous block in reading order, or null")
    p_prev.set_defaults(func=cmd_prev)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
