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
    _dump(_open(args.source).grep(args.pattern, ignore_case=args.ignore_case))
    return 0


def cmd_read_block(args) -> int:
    _dump(_open(args.source).read_block(args.bbox_id))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="strata", description="MinerU retrieval tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = argparse.ArgumentParser(add_help=False)
    source.add_argument("-s", "--source", required=True, help="MinerU `auto` directory")

    p_outline = subparsers.add_parser("outline", parents=[source], help="Title hierarchy (TOC)")
    p_outline.set_defaults(func=cmd_outline)

    p_grep = subparsers.add_parser("grep", parents=[source], help="Regex search over block content")
    p_grep.add_argument("pattern", help="Substring or regex pattern")
    p_grep.add_argument("-i", "--ignore-case", action="store_true", help="Case-insensitive match")
    p_grep.set_defaults(func=cmd_grep)

    p_read = subparsers.add_parser("read-block", parents=[source], help="Full content of one block")
    p_read.add_argument("bbox_id", help="Stable block id, e.g. p0_b6_s0_image_body")
    p_read.set_defaults(func=cmd_read_block)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
