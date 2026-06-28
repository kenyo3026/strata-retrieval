"""QA synthesis over sampled ChunkRecords -- the retrieval-validation feature entry.

Turns a sampled subset of a document's records into a self-validating QA set: for
each chunk an LLM generates a {question, answer}, and because the question was
generated *from* that chunk we already know which bbox_id should answer it -- the
by-construction ground truth, no human labelling (see .prps/qa-synthesis.md).

A manufactured ruler is only trustworthy once validated, so synthesis is two layers:
generate, then qualify each item with an LLM judge. Qualification is two sequential calls:
first the question is bare-answered with no passage in context, then a single judge call
sees {passage, question, answer, bare-answer} and returns both verdicts -- answer_in_chunk
(answer supported by the unit) and answerable_without_doc (bare answer already covered it).
Survivors are the test set.

LLM access aligns with the eval project: litellm underneath, instructor wrapping it for
structured output, models declared as named blocks in a config.yaml read via ConfigMorpher
(`model[name=generator]` / `model[name=qualifier]`). pydantic is confined to the instructor
response_model seam (`_QAResponse`); everything the module hands out is a dataclass.

Synthesizer is a records-in consumer parallel to Sampler/Document. Opening the docs
(retrieval Main) and sampling (Sampler) live in the synthesize() entry, driven by a
`sampling:` spec list in the config -- which doc, draw mode, count, key, generator.
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import pathlib
import random
import shutil
import sys
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Callable, Literal, Optional, Union

import instructor
import litellm
from config_morpher import ConfigMorpher
from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .main import DEFAULT_CHECKPOINT_ROOT, Main
from .prompts import (
    CUSTOM_INSTRUCTIONS_BLOCK,
    INSTRUCTION_FOR_BARE_ANSWER,
    INSTRUCTION_FOR_QA_JUDGE,
    INSTRUCTION_FOR_QA_SYNTHESIS,
)
from .providers.record import ChunkRecord
from .providers.sampler import Sampler, Unit
from .utils.mixin import NameWithLazyDatetime
from .utils.projects import find_project_root


DEFAULT_CONFIG_PATH = find_project_root() / "configs" / "config.yaml"
DEFAULT_OUTPUT_ROOT = find_project_root() / "outputs"


# Non-OpenAI models reject params they don't support; drop them rather than 400.
litellm.drop_params = True


@dataclass
class SynthesisConfig:
    doc_id: str
    synthesis_model: str
    qualifier_model: str
    sampler: str
    sampling_kwargs: dict = field(default_factory = dict)
    custom_instruction: Optional[str] = None   # per-run synthesis refinement; None omits the block
    enable_qualifier: bool = True   # run the LLM judge qualification (bare-answer + one judge call); False keeps every generated item
    min_signal: Optional[str] = None   # drop units the synthesizer rates below this signal; None keeps all
    per_unit: int = 1   # independent synthesis calls per unit, each producing one QA
    cap: Optional[int] = None   # hard ceiling on qualified survivors; stop early once hit
    shuffle_seed: Optional[int] = None   # seed to shuffle sampled units before synthesis; None keeps sampling order

    def setup_models(self, models:dict):
        self.setup_synthesis_model(models)
        self.setup_qualifier_model(models)

    def setup_synthesis_model(self, models:dict):
        if isinstance(self.synthesis_model, str):
            self.synthesis_model = models[self.synthesis_model]

    def setup_qualifier_model(self, models:dict):
        if isinstance(self.qualifier_model, str):
            self.qualifier_model = models[self.qualifier_model]


@dataclass
class QAItem:
    """One synthesized QA anchored to the records of its source unit. `source_ids` are
    the bbox_ids of every record that fed synthesis -- the by-construction retrieval
    ground truth: one id for a chunk unit, the whole span for a page / section unit.
    `signal` is the synthesizer's pre-QA rating of the source unit's worth; the two
    verdicts are the judge's qualification results, each with its one-line reason, and
    `bare_answer` is the no-document attempt the judge weighed. All are kept for auditability.
    `_id` is the stable join key (`{doc_id}#{n}`, n by sampling order) that aligns this item
    across the synthesis output, the hand-run agent output, and the assembled eval set."""
    _id                    : str                     # join key across artifacts: f"{doc_id}#{n}"
    question               : str
    answer                 : str
    signal                 : str                     # synthesizer's low/high rating of the source unit
    source_ids             : list
    doc_id                 : Optional[str]
    answer_in_chunk               : Optional[bool] = None   # judge: answer supported by the unit
    answer_in_chunk_reason        : Optional[str] = None
    answerable_without_doc        : Optional[bool] = None   # judge: bare answer already covered it
    answerable_without_doc_reason : Optional[str] = None
    bare_answer                   : Optional[str] = None    # the no-document attempt fed to the judge


# instructor's structured-output target -- the only pydantic surface. Converted to
# QAItem at the boundary; pydantic never leaves this module. Deliberately has no
# docstring or field descriptions: instructor serializes both into the schema it sends
# the LLM, so they would leak into the prompt. All synthesis guidance lives in the
# instruction template (see .prompts), keeping this class a pure output shape. `signal`
# is first so the model rates the source unit before committing to a question -- the
# field order is a lightweight gate, not post-hoc labelling.
class _QAResponse(BaseModel):
    signal   : Literal["low", "high"]
    question : str
    answer   : str


# The judge's structured-output target, same instructor seam discipline as _QAResponse:
# no docstring, no field descriptions (instructor would serialize them into the schema and
# leak them into the prompt). All judge guidance lives in INSTRUCTION_FOR_QA_JUDGE. Each
# reason precedes its verdict so the model reasons before committing to the boolean.
class _JudgeResponse(BaseModel):
    answer_in_chunk_reason        : str
    answer_in_chunk               : bool
    answerable_without_doc_reason : str
    answerable_without_doc        : bool


def _build_qa_messages(passage: str, custom_instruction: Optional[str] = None) -> list[dict]:
    # The generation prompt: a fixed instruction (role + rubric + prohibitions + examples)
    # as the system message, the passage as the user message. An optional custom_instruction
    # is injected as a refine-not-replace block; absent, the block is dropped entirely.
    block = CUSTOM_INSTRUCTIONS_BLOCK.format(custom_instruction=custom_instruction.strip()) if custom_instruction else ""
    return [
        {"role": "system", "content": INSTRUCTION_FOR_QA_SYNTHESIS.format(custom_instructions=block)},
        {"role": "user", "content": f"Passage:\n{passage}\n\nRate the passage's signal, then write one question and its answer."},
    ]


def _build_bare_answer_messages(question: str) -> list[dict]:
    # The bare-answer probe: the fixed instruction as the system message, the question
    # alone as the user message -- no passage, no custom_instruction. Stage one of
    # qualification; its output is the evidence the judge weighs for answerable_without_doc.
    return [
        {"role": "system", "content": INSTRUCTION_FOR_BARE_ANSWER},
        {"role": "user", "content": f"Question:\n{question}"},
    ]


def _build_judge_messages(
    passage: str,
    question: str,
    answer: str,
    bare_answer: str,
    custom_instruction: Optional[str] = None,
) -> list[dict]:
    # The qualification prompt: the judge instruction as the system message, the pair plus
    # its bare answer as the user message. The bare answer is the evidence for the
    # answerable_without_doc verdict -- it was produced in a separate, passage-free call so
    # "answerable without the document" is a real test, not introspection over visible context.
    block = CUSTOM_INSTRUCTIONS_BLOCK.format(custom_instruction=custom_instruction.strip()) if custom_instruction else ""
    return [
        {"role": "system", "content": INSTRUCTION_FOR_QA_JUDGE.format(custom_instructions=block)},
        {"role": "user", "content": (
            f"Passage:\n{passage}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Bare answer (the question answered with no access to the passage):\n{bare_answer}\n\n"
            "Judge the two checks."
        )},
    ]


def _chunk_text(record: ChunkRecord) -> str:
    # The generatable payload of a chunk: inline text, else an html table string.
    return record.content or record.html or ""


def _unit_text(unit: Unit) -> str:
    # The generatable context of a sampling unit: its records' text joined in reading
    # order. One chunk for a chunk unit, the whole page / section for a many-record one.
    return "\n".join(t for t in (_chunk_text(r) for r in unit.records) if t)


# Ordinal ranks for the synthesizer's source-unit rating, so min_signal can compare.
_SIGNAL_RANK = {"low": 0, "high": 1}


def _is_qualified(item: QAItem, min_signal: Optional[str] = None) -> bool:
    # An item survives when nothing vetoes it: its signal must reach min_signal (None
    # disables the gate), the answer must be in the unit, and the question must not be
    # answerable without the doc. When the judge is off both verdicts stay None, which
    # abstains rather than vetoes -- so with min_signal None and the judge off every item
    # passes.
    if min_signal is not None and _SIGNAL_RANK[item.signal] < _SIGNAL_RANK[min_signal]:
        return False
    return (item.answer_in_chunk is None or item.answer_in_chunk) \
        and (item.answerable_without_doc is None or not item.answerable_without_doc)


class Synthesizer:
    """Generate + qualify, records-in. Stateless beyond its instructor client: the
    per-run model kwargs and the enable_qualifier toggle are passed to run(); owns no doc lifecycle."""

    def __init__(self):
        # acompletion -> instructor returns an AsyncInstructor; create() is awaitable.
        self._client = instructor.from_litellm(litellm.acompletion)

    async def run(
        self,
        samples: list[Unit],
        synthesis_kwargs:dict={},
        qualifier_kwargs:dict={},
        enable_qualifier: bool = True,
        min_signal: Optional[str] = None,
        custom_instruction: Optional[str] = None,
        per_unit: int = 1,
        on_step: Optional[Callable[[QAItem], None]] = None,
        concurrency: int = 8,
        cap: Optional[int] = None,
    ) -> list[QAItem]:
        # Generate per_unit QA per sampled unit (each its own LLM call), qualify each, keep
        # the survivors: the answer must be in the unit and the question must not be
        # answerable without the document. Units run concurrently up to `concurrency` --
        # generate->qualify stays sequential per unit (qualify needs the generated question),
        # but distinct units overlap, so latency-bound LLM calls amortize.
        #
        # cap is a hard ceiling on survivors: once `cap` are kept we stop draining and
        # cancel every still-running / not-yet-started unit, so the run ends without
        # paying for the rest of the samples -- regardless of how many were drawn.
        #
        # on_step fires once per completed unit so the caller can report progress; it
        # sees every drained item, kept or not. Results are drained one at a time in this
        # single coroutine, so on_step never interleaves -- a caller may append / write
        # from it without a lock. Items arrive in completion order.
        sem = asyncio.Semaphore(concurrency)
        kept = []

        async def process(index: int, unit: Unit) -> list[QAItem]:
            async with sem:
                items = []
                for k in range(per_unit):
                    # global, deterministic item number: contiguous over (sampling order, per-unit k),
                    # so the id never depends on completion order or which items survive.
                    n = index * per_unit + k
                    item = await self._generate(unit, n, synthesis_kwargs, custom_instruction)
                    await self._qualify(
                        item,
                        unit,
                        qualifier_kwargs,
                        enable_qualifier=enable_qualifier,
                        custom_instruction=custom_instruction,
                    )
                    items.append(item)
            return items

        tasks = [asyncio.create_task(process(i, unit)) for i, unit in enumerate(samples)]
        try:
            done = False
            for future in asyncio.as_completed(tasks):
                # A unit yields per_unit items; drain one at a time so on_step never
                # interleaves and cap can stop mid-unit.
                for item in await future:
                    if _is_qualified(item, min_signal):
                        kept.append(item)
                    if on_step is not None:
                        on_step(item)
                    if cap is not None and len(kept) >= cap:
                        done = True
                        break
                if done:
                    break
        finally:
            # Cancel the rest (no-op for already-finished tasks) and swallow the
            # resulting CancelledErrors so a capped run exits cleanly.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return kept

    async def _generate(self, unit: Unit, n: int, synthesis_kwargs:dict={}, custom_instruction: Optional[str] = None) -> QAItem:
        resp = await self._client.chat.completions.create(
            **synthesis_kwargs,
            response_model=_QAResponse,
            messages=_build_qa_messages(_unit_text(unit), custom_instruction),
        )
        doc_id = unit.records[0].doc_id
        return QAItem(
            _id=f"{doc_id}#{n}",
            question=resp.question,
            answer=resp.answer,
            signal=resp.signal,
            source_ids=[r.bbox_id for r in unit.records],
            doc_id=doc_id,
        )

    async def _qualify(
        self,
        item: QAItem,
        unit: Unit,
        qualifier_kwargs:dict={},
        enable_qualifier: bool = True,
        custom_instruction: Optional[str] = None,
    ) -> None:

        # enable_qualifier=False disables qualification entirely: both verdicts stay None
        # ("not evaluated") and every generated item survives. With it on, two sequential
        # calls: first bare-answer the question with no passage (the behavioural evidence),
        # then one judge call sees passage + pair + bare answer and returns both verdicts at
        # once. Both run on qualifier_kwargs; the judge reuses the instructor client.
        if not enable_qualifier:
            return

        # Stage one: bare-answer the question with no passage -- free text, so the response
        # model is a plain str.
        item.bare_answer = await self._client.chat.completions.create(
            **qualifier_kwargs,
            response_model=str,
            messages=_build_bare_answer_messages(item.question),
        )

        # Stage two: one judge call sees passage + pair + bare answer, returns both verdicts.
        verdict = await self._client.chat.completions.create(
            **qualifier_kwargs,
            response_model=_JudgeResponse,
            messages=_build_judge_messages(
                _unit_text(unit),
                item.question,
                item.answer,
                item.bare_answer,
                custom_instruction,
            ),
        )
        item.answer_in_chunk = verdict.answer_in_chunk
        item.answer_in_chunk_reason = verdict.answer_in_chunk_reason
        item.answerable_without_doc = verdict.answerable_without_doc
        item.answerable_without_doc_reason = verdict.answerable_without_doc_reason


class SynthesisArtifact(type(pathlib.Path())):
    """One synthesis run's output bundle, addressed like MinerUArtifact / DocCheckpoint.

    Holds two QA files -- `dataset` (qualification survivors, the test set) and its
    `dataset_raw` superset (every generated item with its verdicts, pre-filter) -- plus
    per source doc the full record universe. The record file is keyed by bbox_id, so each
    id in a QAItem.source_ids reverse-looks-up its ref full text (open the doc's records by
    the item's doc_id, index by the source_ids); the drawn units are themselves recoverable
    from dataset_raw's source_ids, so no separate sample file is kept. Pure addressing +
    writes; no run logic.
    """

    @property
    def dataset(self) -> pathlib.Path:
        return self / "dataset.json"

    @property
    def dataset_raw(self) -> pathlib.Path:
        return self / "dataset.raw.json"

    def records(self, doc_id: str) -> pathlib.Path:
        return self / f"{doc_id}_records.json"

    def _dump_by_id(self, records: list[ChunkRecord]) -> str:
        # bbox_id-keyed so each id in dataset's source_ids indexes straight in; a dict
        # preserves insertion (reading) order on py3.7+, so nothing is lost versus a list,
        # and dedups records shared across overlapping units (by-level sections).
        return json.dumps(
            {r.bbox_id: dataclasses.asdict(r) for r in records},
            ensure_ascii=False,
            indent=2,
        )

    def write_records(self, doc_id: str, records: list[ChunkRecord]) -> None:
        self.mkdir(parents=True, exist_ok=True)
        self.records(doc_id).write_text(self._dump_by_id(records), encoding="utf-8")

    def write_dataset(self, qa: list[QAItem]) -> None:
        self.mkdir(parents=True, exist_ok=True)
        self.dataset.write_text(_dump(qa), encoding="utf-8")

    def write_dataset_raw(self, items: list[QAItem]) -> None:
        self.mkdir(parents=True, exist_ok=True)
        self.dataset_raw.write_text(_dump(items), encoding="utf-8")


@dataclass
class SynthesisArgs:
    """Arg schema + parser for the synthesis entry, aligned with the eval project's
    MainArgs: each default reads from `self.<field>` so a subclass overrides in one
    place. Input is an opened-checkpoint doc, not a dataset file."""
    config          : str = str(DEFAULT_CONFIG_PATH)
    source          : Optional[list] = None
    checkpoint      : str = str(DEFAULT_CHECKPOINT_ROOT)
    out_dir         : str = None #str(DEFAULT_OUTPUT_ROOT)
    concurrency     : int = 8
    verbose         : bool = True
    log_level       : str = "ERROR"

    @classmethod
    def from_args(cls) -> "SynthesisArgs":
        args = cls().setup_parser().parse_args()  # instance so defaults read self.<field>
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in args.__dict__.items() if k in field_names})

    def setup_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="strata-synthesis", description="Synthesize a QA set from sampled chunks")
        parser.add_argument("-c", "--config", default=self.config, help="Config YAML with named model blocks and a sampling spec list")
        parser.add_argument(
            "--source",
            action="append",
            metavar="DIR",
            help="MinerU artifact dir to open at startup (repeatable). doc_id defaults to its basename.",
        )
        parser.add_argument(
            "--checkpoint",
            metavar="DIR",
            default=self.checkpoint,
            help="Persist opened docs here; reuse the same dir to inherit them on restart.",
        )
        parser.add_argument("-o", "--out_dir", default=self.out_dir, help="Write the systhesized QA set directory here")
        parser.add_argument("--concurrency", type=int, default=self.concurrency, help="Max records processed concurrently (in-flight LLM pipelines)")
        parser.add_argument("--no-verbose", dest="verbose", action="store_false", default=self.verbose, help="Silence the full-payload print to stdout")
        parser.add_argument("--log-level", default=self.log_level, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level; default ERROR hides LiteLLM warnings")
        return parser

    def __post_init__(self):
        if not self.out_dir and DEFAULT_OUTPUT_ROOT:
            self.out_dir = DEFAULT_OUTPUT_ROOT / str(NameWithLazyDatetime("out"))

        self.out_dir = pathlib.Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)


def _dump(qa: list[QAItem]) -> str:
    return json.dumps([dataclasses.asdict(it) for it in qa], ensure_ascii=False, indent=2)


def _sample(sampler: Sampler, mode: str, **kwargs) -> list[Unit]:
    # Dispatch a sampling spec to the matching Sampler method, which returns the drawn
    # Units -- each one context the generator treats as a single ref, regardless of mode.
    # `key` is a record attribute name turned into the accessor Sampler expects (chunk
    # modes only); other kwargs (k / fraction / level / where) splat straight through.
    draw = getattr(sampler, mode, None)
    supported = {*sampler.CHUNK_BASED_SAMPLING, *sampler.PAGE_BASED_SAMPLING, *sampler.SECTION_BASED_SAMPLING}
    if draw is None or getattr(draw, "__func__", None) not in supported:
        names = sorted(f.__name__ for f in supported)
        raise ValueError(f"unsupported sampling mode {mode!r}; expected one of {names}")
    if key := kwargs.get("key"):
        kwargs["key"] = (lambda r, _k=key: getattr(r, _k))
    return draw(**kwargs)


async def synthesize(args: SynthesisArgs) -> list[QAItem]:
    """Open the sources into the checkpoint, then for each sampling spec in the
    config: sample, generate, and qualify. The spec list -- which doc, draw mode,
    count, stratification key, and generator model -- lives in config.yaml; the
    qualifier model is the fixed `qualifier` block.

    With args.verbose, a rich progress view is rendered to stderr (a per-doc bar
    that ticks once per record with a live kept count), leaving stdout clean for
    the JSON payload. Without it, the run is silent.

    Output is a SynthesisArtifact dir: dataset.raw.json (every generated item) and
    dataset.json (the qualification survivors), both rewritten as they grow so a long
    run is observable and survivable mid-flight, plus, per doc, the full record universe
    that source_ids reverse-look-up into."""
    # stderr so the JSON on stdout stays pipe-clean; disabled => the bar no-ops.
    console = Console(stderr=True)
    artifact = SynthesisArtifact(args.out_dir)

    main = Main(args.checkpoint)
    for source in args.source or []:
        if args.verbose:
            console.print(f"[dim]opening[/] {source}")
        main.open(source)

    config_morpher = ConfigMorpher(args.config)
    models = {
        m["name"]: {k: v for k, v in m.items() if k != "name"}
        for m in config_morpher.fetch("models", [])
    }

    qa: list[QAItem] = []     # qualification survivors -> dataset.json
    raw: list[QAItem] = []    # every generated item -> dataset.raw.json
    specs = config_morpher.fetch("synthesis", [])
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("kept [green]{task.fields[kept]}"),
        TimeElapsedColumn(),
        console=console,
        disable=not args.verbose,
    )
    with progress:
        for syn_config in specs:
            syn_config = SynthesisConfig(**syn_config)
            syn_config.setup_models(models=models)

            records = main.doc(syn_config.doc_id).records
            # The full record universe -- so source_ids can reverse-look-up their ref full
            # text and refs can expand to neighbours later. `where` is no longer a
            # synthesis-side pre-filter: it rides in sampling_kwargs and is applied
            # per-unit inside the sampler (after pages / sections are built).
            artifact.write_records(syn_config.doc_id, records)
            sampler = Sampler(records, syn_config.doc_id)
            samples = _sample(sampler, syn_config.sampler, **syn_config.sampling_kwargs)

            # Shuffle before synthesis so an early `cap` does not bias survivors toward the
            # front of the sampling order (e.g. all from the first pages). Seeded for
            # reproducibility; None leaves the sampler's order untouched.
            if syn_config.shuffle_seed is not None:
                random.Random(syn_config.shuffle_seed).shuffle(samples)

            task = progress.add_task(syn_config.doc_id, total=len(samples), kept=0)
            seen = {"kept": 0}

            # on_step accumulates every item into raw and the survivors into qa, flushing
            # each file as it grows, so accumulation lives here (not via run's return) to
            # keep one writer. run calls it synchronously with no await between, so even
            # under concurrent records it never interleaves -- no lock needed. Both lists
            # are in completion order.
            def on_step(item: QAItem, task=task, seen=seen, min_signal=syn_config.min_signal) -> None:
                raw.append(item)
                artifact.write_dataset_raw(raw)
                if _is_qualified(item, min_signal):
                    seen["kept"] += 1
                    qa.append(item)
                    artifact.write_dataset(qa)
                progress.update(task, advance=1, kept=seen["kept"])

            synthesizer = Synthesizer()
            await synthesizer.run(
                samples,
                syn_config.synthesis_model,
                syn_config.qualifier_model,
                enable_qualifier=syn_config.enable_qualifier,
                min_signal=syn_config.min_signal,
                custom_instruction=syn_config.custom_instruction,
                per_unit=syn_config.per_unit,
                on_step=on_step,
                concurrency=args.concurrency,
                cap=syn_config.cap,
            )

    # Final flush so both files exist even when nothing was generated / kept.
    artifact.write_dataset_raw(raw)
    artifact.write_dataset(qa)

    if args.verbose:
        console.print(f"[green]done[/] generated {len(raw)}, kept {len(qa)} across {len(specs)} docs -> {artifact}")

    return qa


def main() -> int:
    args = SynthesisArgs.from_args()
    logging.getLogger("LiteLLM").setLevel(args.log_level)
    try:
        qa = asyncio.run(synthesize(args))
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # synthesize() owns the file (written incrementally); stdout just mirrors the
    # final payload when verbose.
    if args.verbose:
        print(_dump(qa))

    return 0


if __name__ == "__main__":
    sys.exit(main())
