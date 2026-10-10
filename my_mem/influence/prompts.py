"""最小闭环中的三个 LLM 语义操作。"""

# 候选无关调用：先为所有候选建立同一套比较基准。
TASK_FRAME_PROMPT = """Build a candidate-independent TaskFrame for the current input.

Task:
{task}

Use only information in the task. Do not anticipate a particular memory and do
not claim that the listed dimensions are exhaustive.

Classify query_kind as question, statement, request, or mixed. For a statement,
infer the response obligations naturally created by the statement, such as
acknowledging, explaining, assessing implications, giving advice, or asking a
necessary clarification. Give every obligation a numeric priority. Do not ask
the user merely because several obligations are close in priority; that policy
is reserved for a later version.

Create a small set of common_dimensions shared by every candidate. A core
dimension can change the main conclusion, safety, feasibility, or recommended
action. A secondary dimension changes only preference, tone, or implementation.
The baseline_response is the response outline supported by the task alone,
before using any candidate memory.

Return one JSON object with key `task_frame`. IDs must be unique. Priorities are
JSON numbers from 0.0 to 1.0. Follow the JSON Schema below exactly.
"""

# 第一次调用：原生判断 high/low/none，并只为真正缺失的桥接事实提出 gap。
HYPOTHESIS_PROMPT = """You analyze how strongly each retrieved memory should influence the
answer, judgment, or action that the current task naturally calls for. Judge the
task's focal decision, not any conceivable side topic or merely possible reply.

Task:
{task}

Candidate-independent TaskFrame:
{task_frame_json}

Candidate memories:
{candidates_json}

For every candidate, return exactly one hypothesis using this shared rubric:

- high: the memory supplies a constraint that materially changes the core
  answer, final choice, safety/feasibility priority, or recommended action.
- low: the memory has a justified connection and makes only a minor adjustment
  to preference, tone, experience, or implementation; it cannot dominate the
  core decision.
- none: the answer and action should not change. Lexical/topic similarity,
  analogy, an alternative story, or an unproven assumption that two events are
  identical is not influence.

Use `impact` only to rank candidates after assigning the level. Keep the fields
consistent: none uses 0.0 <= impact < 0.20; low uses 0.20 <= impact < 0.60; high
uses 0.60 <= impact <= 1.0. A none hypothesis has no gaps, no bridge_steps, no
proposed_dimension, and no possible_action_change. A low or high hypothesis
must state the corresponding minor or material possible_action_change.

First try to map each candidate to zero or more common dimension IDs in
`matched_dimension_ids`. Mentioning a dimension does not by itself establish
influence, so a none candidate may still map to a common dimension. A low or
high candidate must target at least one common dimension or propose exactly one
candidate-local `proposed_dimension`. A none candidate cannot propose a new
dimension.

Use `proposed_dimension` only when the fixed TaskFrame genuinely omitted a
material dimension. It must quote or closely identify the grounding in the
memory, state the minimal bridge to the query, and describe the decision delta.
It is temporary for this candidate and this task: do not modify the supplied
TaskFrame, do not reuse it for another candidate, and do not request a second
candidate-recall round. If no new dimension is needed, use null.

First separate facts stated by the task (`task_supported_claims`) and facts
stated by the memory (`memory_supported_claims`) from inferred bridge claims.
Never create a gap for a fact already stated by the task or memory. A gap is
allowed only for a missing inferred bridge that can change the influence level,
and it must be a concrete proposition that evidence could support or refute.
Retrieval metadata is not evidence.

Express the complete influence bridge once in `bridge_steps`. Mark every step's
basis as task, memory, inference, or gap. A gap-based step must cite a gap_id.
`bridge_confidence` is the confidence that the bridge as a whole holds, not the
size of its potential effect. Do not make another model call to expand a long
bridge.

`source_type` describes where to obtain missing evidence, never the semantic
topic. Use only: personal_memory_store for additional personal memories;
external_knowledge for stable public knowledge; environment_tool for current
environment, records, or tools; user for information that only the user can
clarify. Facts already supplied by the task or candidate are supported claims,
not retrieval sources and not gaps. Do not invent domain-specific source types.

Return one JSON object with key `hypotheses`. Each hypothesis must contain:
candidate_id, influence_level (high|low|none), effect_target,
matched_dimension_ids, proposed_dimension, mechanism, bridge_steps, conditions,
task_supported_claims, memory_supported_claims, inferred_claims, gaps,
possible_action_change, impact, bridge_confidence. Each gap must contain gap_id,
proposition, source_type, retrieval_query, decision_relevance, priority, status
(use `open`). Array fields must be JSON arrays. `impact`, `bridge_confidence`,
bridge-step confidence, dimension priority, and gap priority are JSON numbers
from 0.0 to 1.0, never labels such as "high" or "medium". Follow the JSON Schema
below exactly.
"""

# 第二次调用：重新看到原始记忆和实际取得的证据，再决定接受、拒绝或保留未知。
VALIDATION_PROMPT = """Validate a proposed memory influence using only the supplied evidence.

Task:
{task}

Original candidate memory:
{candidate_json}

Hypothesis:
{hypothesis_json}

Acquired evidence:
{evidence_json}

Do not use retrieval metadata or unsupported model knowledge as evidence. The
original candidate may be cited as `memory:<candidate_id>`; acquired evidence
must be cited by its supplied evidence_id.
Return one JSON object with key `result`, containing candidate_id, status
(validated|rejected|irrelevant|unresolved), influence_level (high|low|none),
conclusion, evidence_ids, unresolved_gap_ids, action_implication and confidence.
A rejected or irrelevant result must use influence_level=none. A validated
result must use high or low and cite at least one supplied evidence_id. An
unresolved result retains the best provisional high or low level while listing
the unresolved gaps. `confidence` is a JSON number from 0.0 to 1.0. Follow the
JSON Schema below exactly.
"""
