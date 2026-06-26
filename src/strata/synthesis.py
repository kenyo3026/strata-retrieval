"""QA synthesis over sampled ChunkRecords -- the retrieval-validation feature entry.

Turns a sampled subset of a document's records into a self-validating QA set: for
each chunk an LLM generates a {question, answer}, and because the question was
generated *from* that chunk we already know which bbox_id should answer it -- the
by-construction ground truth, no human labelling (see .prps/qa-synthesis.md).

A manufactured ruler is only trustworthy once validated, so synthesis is two layers:
generate, then qualify each item as a valid anchor (3a answer-in-chunk, deterministic;
3b need-context, a behavioural LLM bare-answer test). Survivors are the test set.

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
import shutil
import re
import sys
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Callable, ClassVar, Optional, Union

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
from .providers.record import ChunkRecord
from .providers.sampler import PageSample, Sampler
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
    """One synthesized QA anchored to its source chunk. `source_id` == the
    originating ChunkRecord.bbox_id -- the by-construction retrieval ground truth.
    The two verdicts are step-3 qualification results, kept for auditability."""
    question               : str
    answer                 : str
    source_id              : str
    doc_id                 : Optional[str]
    answer_in_chunk        : Optional[bool] = None   # 3a: answer located in the chunk
    answerable_without_doc : Optional[bool] = None   # 3b: answerable from bare knowledge


# instructor's structured-output target -- the only pydantic surface. Converted to
# QAItem at the boundary; pydantic never leaves this module. Kept as a comment, not a
# docstring, on purpose: instructor sends a model's docstring + field descriptions to
# the LLM, so the docstring would leak into the prompt.
class _QAResponse(BaseModel):
    question : str
    answer   : str

    # The generation prompt is coupled to this output shape, so the whole message template
    # lives with the schema rather than as free module constants. ClassVar keeps it out of
    # the JSON schema, so instructor's response is unaffected.
    messages: ClassVar[list[dict]] = [
        {
            "role": "system",
            "content": (
                "You generate one question-answer pair from a passage extracted from a document. "
                "The question must be answerable using only the passage and specific enough that it "
                "points to this passage rather than general knowledge. The answer must be a concise, "
                "factual statement supported by the passage. Do not ask about the passage's location, "
                "figure or table numbers, or formatting."
            )
        },
        {
            "role": "user",
            "content": "Passage:\n{chunk}\n\nGenerate one question and its answer."
        },
    ]

    @classmethod
    def build_messages(cls, chunk: str) -> list[dict]:
        # Fresh list with each content formatted; never mutate the shared class template.
        return [{**m, "content": m["content"].format(chunk=chunk)} for m in cls.messages]


def _chunk_text(record: ChunkRecord) -> str:
    # The generatable payload of a chunk: inline text, else an html table string.
    return record.content or record.html or ""


def _tokens(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def _overlap(part: str, whole: str) -> float:
    # Fraction of `part`'s tokens present in `whole`. Robust to the LLM rephrasing,
    # unlike substring matching; 0.0 when `part` has no tokens.
    pt = _tokens(part)
    if not pt:
        return 0.0
    return len(pt & _tokens(whole)) / len(pt)


# The bare-answer prompt has no response schema to bind to (3b returns free text via a
# plain completion), so it stays a module constant owned by the qualify step.
_BARE_SYSTEM = (
    "Answer the question concisely from your own knowledge. "
    "If you do not know the answer, say you do not know."
)


class Synthesizer:
    """Generate + qualify, records-in. Holds two resolved litellm model kwargs
    (generator / qualifier) and the two overlap thresholds; owns no doc lifecycle."""

    def __init__(
        self,
        answer_overlap: float = 0.6,
        bare_overlap: float = 0.6,
    ):
        self.answer_overlap = answer_overlap
        self.bare_overlap = bare_overlap

        # acompletion -> instructor returns an AsyncInstructor; create() is awaitable.
        self._client = instructor.from_litellm(litellm.acompletion)

    async def run(
        self,
        samples: list[ChunkRecord],
        synthesis_kwargs:dict={},
        qualifier_kwargs:dict={},
        answer_overlap: float = 0.6,
        bare_overlap: float = 0.6,
        on_step: Optional[Callable[[QAItem], None]] = None,
        concurrency: int = 8,
    ) -> list[QAItem]:
        # Generate one QA per sampled chunk, qualify each, keep the survivors: the
        # answer must be in the chunk (3a) and the question must not be answerable
        # without the document (3b). Records run concurrently up to `concurrency` --
        # generate->qualify stays sequential per record (qualify needs the generated
        # question), but distinct records overlap, so latency-bound LLM calls amortize.
        #
        # on_step fires once per record (after both LLM calls) so the caller can report
        # progress; it sees every item, kept or not. It is called synchronously and must
        # not await -- on a single event loop that makes it atomic, so a caller may
        # append / write from it without a lock. Items arrive in completion order.
        sem = asyncio.Semaphore(concurrency)
        kept = []

        async def process(record: ChunkRecord) -> None:
            async with sem:
                item = await self._generate(record, synthesis_kwargs)
                await self._qualify(
                    item,
                    record,
                    qualifier_kwargs,
                    answer_overlap=answer_overlap,
                    bare_overlap=bare_overlap,
                )
            if item.answer_in_chunk and not item.answerable_without_doc:
                kept.append(item)
            if on_step is not None:
                on_step(item)

        await asyncio.gather(*(process(record) for record in samples))
        return kept

    async def _generate(self, record: ChunkRecord, synthesis_kwargs:dict={}) -> QAItem:
        resp = await self._client.chat.completions.create(
            **synthesis_kwargs,
            response_model=_QAResponse,
            messages=_QAResponse.build_messages(_chunk_text(record)),
        )
        return QAItem(
            question=resp.question,
            answer=resp.answer,
            source_id=record.bbox_id,
            doc_id=record.doc_id,
        )

    async def _qualify(
        self,
        item: QAItem,
        record: ChunkRecord,
        qualifier_kwargs:dict={},
        answer_overlap: Optional[float] = None,
        bare_overlap: Optional[float] = None,
    ) -> None:

        answer_overlap = answer_overlap or self.answer_overlap
        bare_overlap = bare_overlap or self.bare_overlap

        # 3a: deterministic -- is the answer's content actually in the source chunk.
        item.answer_in_chunk = _overlap(item.answer, _chunk_text(record)) >= answer_overlap

        # 3b: behavioural -- let the model answer the question with no document; if its
        # bare answer matches the gold answer, the question doesn't exercise retrieval.
        bare = await self._bare_answer(item.question, qualifier_kwargs)
        item.answerable_without_doc = _overlap(bare, item.answer) >= bare_overlap

    async def _bare_answer(self, question: str, qualifier_kwargs:dict={}) -> str:
        resp = await litellm.acompletion(
            **qualifier_kwargs,
            messages=[
                {"role": "system", "content": _BARE_SYSTEM},
                {"role": "user", "content": question},
            ],
        )
        return resp.choices[0].message.content or ""


class SynthesisArtifact(type(pathlib.Path())):
    """One synthesis run's output bundle, addressed like MinerUArtifact / DocCheckpoint.

    Holds two QA files -- `dataset` (qualification survivors, the test set) and its
    `dataset_raw` superset (every generated item with its verdicts, pre-filter) -- plus
    per source doc the full record universe and the sampled subset that fed generation.
    The two record files are keyed by bbox_id, so a QAItem.source_id reverse-looks-up its
    ref full text (open the doc's records by the item's doc_id, index by source_id). Pure
    addressing + writes; no run logic.
    """

    @property
    def dataset(self) -> pathlib.Path:
        return self / "dataset.json"

    @property
    def dataset_raw(self) -> pathlib.Path:
        return self / "dataset.raw.json"

    def records(self, doc_id: str) -> pathlib.Path:
        return self / f"{doc_id}_records.json"

    def sample_records(self, doc_id: str) -> pathlib.Path:
        return self / f"{doc_id}_sample_records.json"

    def _dump_by_id(self, records: list[ChunkRecord]) -> str:
        # bbox_id-keyed so dataset's source_id indexes straight in; a dict preserves
        # insertion (reading) order on py3.7+, so nothing is lost versus a list.
        return json.dumps(
            {r.bbox_id: dataclasses.asdict(r) for r in records},
            ensure_ascii=False,
            indent=2,
        )

    def write_records(self, doc_id: str, records: list[ChunkRecord]) -> None:
        self.mkdir(parents=True, exist_ok=True)
        self.records(doc_id).write_text(self._dump_by_id(records), encoding="utf-8")

    def write_sample_records(self, doc_id: str, samples: list[ChunkRecord]) -> None:
        self.mkdir(parents=True, exist_ok=True)
        self.sample_records(doc_id).write_text(self._dump_by_id(samples), encoding="utf-8")

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


def _match(record: ChunkRecord, where: dict) -> bool:
    # A record passes when every attr satisfies its condition: membership for a
    # list/set value, equality otherwise. The WHERE that runs before sampling --
    # which records are eligible -- orthogonal to `key`, which only stratifies.
    for attr, cond in where.items():
        val = getattr(record, attr)
        ok = val in cond if isinstance(cond, (list, set)) else val == cond
        if not ok:
            return False
    return True


def _dump(qa: list[QAItem]) -> str:
    return json.dumps([dataclasses.asdict(it) for it in qa], ensure_ascii=False, indent=2)


def _sample(sampler: Sampler, mode: str, **kwargs) -> list[ChunkRecord]:
    # # Dispatch a sampling spec to the matching Sampler method. `key` is a record
    # # attribute name turned into the accessor Sampler expects, and applies only to
    # # chunk modes; page modes have no stratification key and come back as whole
    # # PageSamples, which we flatten to records so QA stays chunk (bbox_id) granular.
    draw = getattr(sampler, mode)

    if draw.__func__ in sampler.CHUNK_BASED_SAMPLING:
        if key:=kwargs.get("key"):
            kwargs["key"] = (lambda r, _k=key: getattr(r, _k)) if key else None
        samples = draw(**kwargs)

    elif draw.__func__ in sampler.PAGE_BASED_SAMPLING:
        samples = draw(**kwargs)
        if samples and isinstance(samples[0], PageSample):
            samples = [record for page in samples for record in page.records]

    else:
        supported = sorted(f.__name__ for f in (*sampler.CHUNK_BASED_SAMPLING, *sampler.PAGE_BASED_SAMPLING))
        raise ValueError(f"unsupported sampling mode {mode!r}; expected one of {supported}")

    return samples


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
    and the sampled subset that fed generation."""
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
    generated = 0
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
            # The full record universe, before the where filter -- so source_id can
            # reverse-look-up its ref full text and refs can expand to neighbours later.
            artifact.write_records(syn_config.doc_id, records)
            # pop, not read: `where` is a synthesis-side filter, not a Sampler param,
            # so it must leave sampling_kwargs before the **splat into _sample.
            where = syn_config.sampling_kwargs.pop("where", None)
            if where:
                records = [r for r in records if _match(r, where)]
            sampler = Sampler(records, syn_config.doc_id)
            samples = _sample(sampler, syn_config.sampler, **syn_config.sampling_kwargs)
            # The sampled subset that actually fed generation -- the draw's audit trail.
            artifact.write_sample_records(syn_config.doc_id, samples)
            generated += len(samples)

            task = progress.add_task(syn_config.doc_id, total=len(samples), kept=0)
            seen = {"kept": 0}

            # on_step accumulates every item into raw and the survivors into qa, flushing
            # each file as it grows, so accumulation lives here (not via run's return) to
            # keep one writer. run calls it synchronously with no await between, so even
            # under concurrent records it never interleaves -- no lock needed. Both lists
            # are in completion order.
            def on_step(item: QAItem, task=task, seen=seen) -> None:
                raw.append(item)
                artifact.write_dataset_raw(raw)
                if item.answer_in_chunk and not item.answerable_without_doc:
                    seen["kept"] += 1
                    qa.append(item)
                    artifact.write_dataset(qa)
                progress.update(task, advance=1, kept=seen["kept"])

            synthesizer = Synthesizer()
            await synthesizer.run(
                samples,
                syn_config.synthesis_model,
                syn_config.qualifier_model,
                on_step=on_step,
                concurrency=args.concurrency,
            )

    # Final flush so both files exist even when nothing was generated / kept.
    artifact.write_dataset_raw(raw)
    artifact.write_dataset(qa)

    if args.verbose:
        console.print(f"[green]done[/] generated {generated}, kept {len(qa)} across {len(specs)} docs -> {artifact}")

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
