"""Format a synthesis output dir into the jsonl the evals engine consumes.

Synthesis produces the gold QA set (dataset.json) plus the per-doc record universe; this
turns them into one evals-ready jsonl whose columns are the LLMTestCase fields the judge
reads -- input / expected_output / retrieval_context / actual_output. ref text is
reconstituted from each item's source_ids via SynthesisArtifact.ref_text (the compressed
dataset.json carries only the ids).

actual_output -- the agent's final answer -- is foreign to the synthesis artifact: it is
produced by a separate, hand-run agent and joined back here via -a (a jsonl keyed by _id).
Without -a, actual_output is null, so the file can be inspected before the agent run.

No DeepEval dependency lives here: the judge runs in the separate evals engine, and the two
meet only at this jsonl. strata never imports the judge; the engine never imports the agent.
"""

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass, fields
from typing import Optional

from .synthesis import SynthesisArtifact


@dataclass
class EvalCase:
    """One evals-engine test case, serialized one per line into the evalset jsonl. Field names
    are fixed by DeepEval's LLMTestCase columns (input / expected_output / retrieval_context /
    actual_output) -- the external contract dictates them, so they are not renamed to dodge the
    `input` builtin shadow. `_id` is not a judge column; it rides along for traceability and the
    agent-output join."""
    _id               : str
    input             : str            # the question the agent is asked
    expected_output   : str            # gold answer A -- AnswerCoverage's yardstick
    retrieval_context : list           # [ref] source text -- SourceFaithfulness's yardstick
    actual_output     : Optional[str] = None   # agent's final answer; None until joined via -a


def _read_answers(path: pathlib.Path) -> dict:
    # Agent output: one json object per line, each at least {_id, actual_output}. Reduced to
    # _id -> actual_output for the join; lines without an _id can't be matched, so are skipped.
    answers = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "_id" in row:
            answers[row["_id"]] = row.get("actual_output")
    return answers


def build_evalset(artifact_dir, answer_path: Optional[str] = None) -> list[EvalCase]:
    # dataset.json (survivors) -> one EvalCase per item. retrieval_context is a single-element
    # list ([ref]) because DeepEval treats it as a list of context chunks. actual_output joins
    # from the agent output by _id; absent -> None.
    art = SynthesisArtifact(artifact_dir)
    items = json.loads(art.dataset.read_text(encoding="utf-8"))
    answers = _read_answers(pathlib.Path(answer_path)) if answer_path else {}
    return [
        EvalCase(
            _id=it["_id"],
            input=it["question"],
            expected_output=it["answer"],
            retrieval_context=[art.ref_text(it["doc_id"], it["source_ids"])],
            actual_output=answers.get(it["_id"]),
        )
        for it in items
    ]


def _dump_jsonl(rows: list[EvalCase], path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


@dataclass
class FormatEvalsetArgs:
    """Arg schema + parser for strata-format-evalset, same MainArgs-style pattern as
    SynthesisArgs: each default reads from `self.<field>` so a subclass overrides in one place."""
    artifact : str = None
    answer   : Optional[str] = None
    out      : Optional[str] = None

    @classmethod
    def from_args(cls) -> "FormatEvalsetArgs":
        args = cls().setup_parser().parse_args()  # instance so defaults read self.<field>
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in args.__dict__.items() if k in field_names})

    def setup_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="strata-prepare-evalset", description="Prepare an evals-ready jsonl from a synthesis output dir")
        parser.add_argument("-d", "--artifact", required=not self.artifact, default=self.artifact, help="Synthesis output dir (holds dataset.json + {doc_id}_records.json)")
        parser.add_argument("-a", "--answer", default=self.answer, help="Agent output jsonl keyed by _id; omit to leave actual_output null")
        parser.add_argument("-o", "--out", default=self.out, help="Output jsonl path (default: eval_dataset.jsonl beside -a if given, else in the -d dir)")
        return parser

    def __post_init__(self):
        if not self.artifact:  # the bare cls() instance used only to read parser defaults
            return
        self.artifact = pathlib.Path(self.artifact)
        if self.out:
            self.out = pathlib.Path(self.out)
        elif self.answer:
            # default beside the agent output: the evalset belongs to that specific agent run
            self.out = pathlib.Path(self.answer).parent / "eval_dataset.jsonl"
        else:
            self.out = self.artifact / "eval_dataset.jsonl"


def main() -> int:
    args = FormatEvalsetArgs.from_args()
    try:
        rows = build_evalset(args.artifact, args.answer)
        _dump_jsonl(rows, args.out)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    with_answer = sum(row.actual_output is not None for row in rows)
    print(f"wrote {len(rows)} rows ({with_answer} with actual_output) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
