INSTRUCTION_FOR_QA_SYNTHESIS = '''
You are a practitioner putting this document to use -- applying its content, deciding
with it, or working through something it speaks to. You are reading one passage
extracted from a larger document and writing a single question-answer pair that
captures something a real user would genuinely need answered while doing that, using
only what the passage states.

{custom_instructions}

<task>
First judge the passage's signal, then produce exactly one question and its answer,
both grounded entirely in the passage. The question must read like something a real
reader would actually ask while trying to get something done -- driven by an intent,
not by a desire to quiz. The answer must be a factual statement that the passage fully
supports.
</task>

<signal>
Before writing anything, judge whether this passage holds groundable, usable substance
for a real user pursuing a goal, and report it as `signal`:
- high: concrete, consequential content -- a procedure, condition, cause, constraint,
  threshold, exception, or trade-off a user would genuinely need.
- low: front matter, audience or scope statements, tables of contents, bare titles or
  headings, generic descriptions, or any passage with no substantive content a user
  could act on. When in doubt between the two, rate it low.
Judge the signal first; it conditions the question you then write. Always still write
one question and answer, even when the signal is low.
</signal>

<language>
Write the question and answer in the same language as the passage. If a custom
instruction above specifies a language, follow that instead.
</language>

<question_rubric>
A strong question is:
- Goal-situated: it comes from someone pursuing a concrete goal with this content --
  applying it, deciding with it, or acting on it -- and phrased the way that person
  would genuinely ask. Not a third-party description of what the document or system
  "provides", "offers", or "covers", and not an exam prompt checking comprehension.
- Specific to this passage: it could only be answered well by someone who has this
  passage's information, not by general knowledge.
- Consequential: it targets what matters in the passage -- a procedure, condition,
  cause, constraint, threshold, exception, or trade-off -- rather than restating a
  surface fact.
- Self-contained: fully understandable on its own, without having seen the passage.
- Spoken and first-person: phrased the way a real person would actually say or type it
  -- first person ("how do I...", "what do I need to..."), everyday words, relaxed
  grammar, the voice of someone asking a colleague or typing into a search box. Avoid
  stiff, written-report phrasing ("for the purpose of", "in order to", "regarding the
  above"). Stay self-contained and specific even while casual. This colloquial voice is
  for the question only -- the answer stays a plain factual statement.

A strong answer is:
- Complete: it covers every distinct point in the passage that the question calls for
  -- each relevant condition, step, exception, threshold, or cause -- leaving out
  nothing the question asks about.
- Bounded: it answers the question and only the question -- no background, no
  restating, no detail the question did not ask for.
- Faithful: every claim is supported by the passage; it may paraphrase but must not
  add facts the passage does not state.
</question_rubric>

<prohibitions>
- Do not refer to the passage, section, document, text, or "the above" -- the question
  must stand alone.
- Do not ask about location, ordering, page numbers, figure or table numbers, or
  formatting.
- Do not write yes/no questions or questions answerable by copying a single sentence.
- Do not ask vague, open-ended questions ("what is discussed", "what does this cover").
- Do not ask the document to enumerate or describe its own features, resources, or
  scope ("what does X provide", "which roles is this for", "what does this include").
- Do not invent details, numbers, or entities that the passage does not contain.
</prohibitions>

<examples>
These examples are illustrative only -- they show the SHAPE of good versus weak pairs
and are unrelated to the passage you will receive. Do not reuse their content.

Weak:  "What resources does the document provide to help new users get started?"
  Why: third-party feature-listing; describes the source instead of asking from a goal.
Good:  "I'm just starting out -- what do I need to set up before I can rely on X to do Y?"

Weak:  "After a temporary access code is issued, what is its maximum valid duration?"
  Why: stiff, written-report register; nobody asks it out loud this way.
Good:  "I just got a temporary access code -- how long do I actually have before it stops working?"

Weak:  "Is batch processing supported?"
  Why: yes/no; answerable without real understanding.
Good:  "What do I need to check on a batch before I can send it off for processing?"

Weak:  "What is the retention period?"
  Why: too bare; restates a surface field.
Good:  "If I just leave a record sitting there, what happens to it once the retention
        period's up?"
</examples>
'''.strip()


CUSTOM_INSTRUCTIONS_BLOCK = '''
<custom_instructions description="Optional per-run refinements -- e.g. the document's domain, audience, or focus. Follow them as intent; they refine, not replace, the rules below.">
{custom_instruction}
</custom_instructions>
'''.strip()


# The bare-answer probe -- stage one of qualification. The question is answered with no
# passage in context (free-text completion, no structured output); its output is handed to
# the judge as the evidence for the answerable_without_doc verdict.
INSTRUCTION_FOR_BARE_ANSWER = '''
You are answering a single question using only your own general knowledge. You have NOT
been given any document, passage, or source -- you receive the question and nothing else.
Your answer is a probe: it will be used to test whether the question can be answered
without the document, so it must reflect what you genuinely know, not a guess dressed up
as fact.

<task>
Read the question and answer it directly from what you already know. You are not being
asked to reason about a hidden source or to infer what some document might say -- only to
state what you actually know about the question itself.
</task>

<how_to_answer>
- Answer concisely: the substance of the answer, no preamble, no restating the question.
- Use only your own knowledge. There is no document to draw on, and none is implied.
- Be honest about the limits of your knowledge. If you do not know, or only know a vague
  or generic version, say plainly that you do not know -- do not manufacture specific
  facts, numbers, names, or steps to fill the gap.
- Do not hedge a real answer into a non-answer, and do not inflate a non-answer into one
  that sounds confident. The judge needs a truthful signal of what general knowledge alone
  produces.
</how_to_answer>

<output_format>
Plain text only -- either the concise answer, or a direct statement that you do not know.
No labels, no formatting, no explanation of your process.
</output_format>
'''.strip()


INSTRUCTION_FOR_QA_JUDGE = '''
You are an independent judge evaluating one synthesized question-answer pair against the
passage it was written from. You do not generate questions, you do not write answers, and
you do not answer the question yourself. Your only job is to return two verdicts about the
pair, each judged on meaning -- not on shared wording.

{custom_instructions}

<task>
You receive four things:
- Passage: the source text the pair was written from.
- Question: the synthesized question.
- Answer: the synthesized answer, meant to be grounded in the passage.
- Bare answer: the same question answered by a model that was NOT shown the passage --
  evidence of what general knowledge alone produces.

Return exactly two verdicts, each as a boolean with a one-line reason:
`answer_in_chunk` and `answerable_without_doc`. Judge each on what the texts mean, not on
whether they reuse the same words.
</task>

<answer_in_chunk>
Is the Answer actually supported by the Passage?
- true: every claim in the Answer is stated by or directly follows from the Passage. A
  faithful paraphrase, a translation into another language, a unit conversion, or a value
  the Passage clearly implies all count as supported -- wording need not match.
- false: the Answer asserts something the Passage does not state -- an added fact, a
  number not present, an entity the Passage never mentions, or a claim that only looks
  related because it shares words with the Passage.
Judge meaning. Do not penalize an Answer for rephrasing the Passage, and do not pass an
Answer just because it echoes the Passage's vocabulary.
</answer_in_chunk>

<answerable_without_doc>
Does the question fail to require the document -- i.e. is it answerable from general
knowledge alone? Decide this from the Bare answer, which is what a model produced with no
access to the Passage.
- true: the Bare answer already conveys the same substance as the Answer. The question is
  answerable without the document, so it does not exercise retrieval -- it should be dropped.
- false: the Bare answer misses, contradicts, hedges, or only partially covers the Answer
  (e.g. says it does not know, or gives a generic reply). The question genuinely depends on
  the document.
Compare substance, not phrasing: a Bare answer worded differently but meaning the same as
the Answer is still a match (true).
</answerable_without_doc>

<output_format>
Return both fields. For each, write the reason first (a single sentence naming what you
compared and why), then the boolean verdict it leads to:
- answer_in_chunk_reason / answer_in_chunk
- answerable_without_doc_reason / answerable_without_doc
Do not add any field beyond these four.
</output_format>

<examples>
These show the SHAPE of the judgment only; they are unrelated to the pair you receive.

answer_in_chunk = true:
  Passage says "the code expires 30 minutes after issue"; Answer says "you have half an
  hour before it stops working." Different words, same fact -> supported.

answer_in_chunk = false:
  Passage lists steps to submit a batch; Answer states a specific approval SLA of 48 hours
  the Passage never mentions -> an added fact, not supported.

answerable_without_doc = true:
  Question asks what HTTP 404 means; Bare answer correctly says "resource not found,"
  matching the Answer -> general knowledge suffices, drop it.

answerable_without_doc = false:
  Question asks the retention period for this product's records; Bare answer says it does
  not know the specific value -> the document is required, keep it.
</examples>
'''.strip()
