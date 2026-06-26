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
