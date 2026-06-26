INSTRUCTION_FOR_QA_SYNTHESIS = '''
You are a practitioner consulting this document to accomplish a real task. You are
reading one passage extracted from a larger document and writing a single
question-answer pair that a real reader would genuinely need, using only what the
passage states.
{custom_instructions}
<task>
Produce exactly one question and its answer, both grounded entirely in the passage.
The question must read like something a real reader would actually ask while trying
to get something done -- driven by an intent, not by a desire to quiz. The answer
must be a concise, factual statement that the passage fully supports.
</task>

<question_rubric>
A strong question is:
- Intent-driven: phrased the way a person with a real goal would ask it, not as a
  textbook or exam prompt.
- Specific to this passage: it could only be answered well by someone who has this
  passage's information, not by general knowledge.
- Consequential: it targets what matters in the passage -- a procedure, condition,
  cause, constraint, threshold, exception, or trade-off -- rather than restating a
  surface fact.
- Self-contained: fully understandable on its own, without having seen the passage.
- Natural: phrased in plain, fluent language.

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
- Do not invent details, numbers, or entities that the passage does not contain.
</prohibitions>

<examples>
These examples are illustrative only -- they show the SHAPE of good versus weak pairs
and are unrelated to the passage you will receive. Do not reuse their content.

Weak:  "What does this section describe?"
  Why: meta and vague; refers to the document and tests nothing specific.
Good:  "After a temporary access code is issued, how long can it still be used before
        it stops working?"

Weak:  "Is batch processing supported?"
  Why: yes/no; answerable without real understanding.
Good:  "What has to be true about a batch before it can be submitted for processing?"

Weak:  "What is the retention period?"
  Why: too bare; restates a surface field.
Good:  "If a record is left untouched, what happens to it once the retention period
        elapses?"
</examples>
'''.strip()


CUSTOM_INSTRUCTIONS_BLOCK = '''
<custom_instructions description="Optional per-run refinements -- e.g. the document's domain, audience, or focus. Follow them as intent; they refine, not replace, the rules below.">
{custom_instruction}
</custom_instructions>
'''.strip()
