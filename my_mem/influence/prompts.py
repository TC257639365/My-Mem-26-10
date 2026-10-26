"""最小闭环中的 LLM 语义操作。"""

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
List open_dimensions as short descriptions of important categories that may be
missing from this preliminary frame. The baseline_response must be a short
response skeleton (obligations and default stance), not a complete answer. Do
not inject medical, legal, moral, or other world knowledge not stated by the
task. Candidate memories are assessed only after a separate dimension challenge.

Return one JSON object with key `task_frame`. IDs must be unique. Priorities are
JSON numbers from 0.0 to 1.0. Follow the JSON Schema below exactly.
"""

# 在影响定级前，所有候选先获得同等的公共维度映射/挑战机会。
DIMENSION_CHALLENGE_PROMPT = """Review how each candidate memory could map to or challenge
the preliminary TaskFrame. Do this before assigning any influence level.

Task:
{task}

Preliminary candidate-independent TaskFrame (F0):
{task_frame_json}

Candidate memories:
{candidates_json}

For every candidate, return exactly one dimension_challenge. First map any
applicable common dimension IDs. If the memory exposes a grounded decision
dimension missing from F0, it may propose exactly one candidate-local dimension,
even when the bridge is uncertain and will later need evidence. A proposal must
identify memory grounding, the minimal query bridge, and the conditional response
or action delta. Do not decide high/low/none here. Do not reject a proposal merely
because identity, event linkage, or another bridge premise is still unverified;
that uncertainty belongs to bridge_status and EvidenceGap in the next stage.

If the candidate neither maps to F0 nor supports a grounded new dimension, leave
both mapping fields empty and explain why in rationale. A proposed dimension is
temporary and does not mutate F0 or trigger candidate recall.

Return one JSON object with key `dimension_challenges`. Follow the JSON Schema
below exactly.
"""

# 维度审查后再分别判断条件影响上界和桥接确定性。
HYPOTHESIS_PROMPT = """Construct one influence hypothesis per candidate after its dimension
challenge has already been reviewed. Keep conditional impact separate from
whether the bridge is currently established.

Task:
{task}

Candidate-independent TaskFrame:
{task_frame_json}

Candidate memories:
{candidates_json}

Reviewed candidate-local dimension challenges:
{dimension_challenges_json}

For every candidate, return exactly one hypothesis using four separate states:

1. `potential_impact`: the effect IF the minimal bridge is true.
   - high: materially changes the core answer, final choice, safety/feasibility
     priority, or recommended action.
   - low: makes only a minor change to preference, tone, explanation, or
     implementation.
   - none: even the strongest grounded interpretation produces no response or
     action delta.
2. `bridge_status`: supported when every material bridge premise is already
   grounded in the task/memory; open when a material, testable premise is
   unresolved; refuted when the required bridge is contradicted or no grounded
   bridge exists.
3. `bridge_confidence`: confidence in the bridge, independent of effect size.
4. `validated_impact` is NOT produced here; it is assigned only after evidence
   acquisition by the validator.

Copy `target_dimension_ids` only from the candidate's reviewed common dimension
IDs or its reviewed proposed dimension ID. Do not create dimensions here. An
open bridge may still have high potential impact. Identity or event uncertainty
must lower bridge confidence and create a gap, not lower potential impact.

First separate facts stated by the task (`task_supported_claims`) and facts
stated by the memory (`memory_supported_claims`) from inferred bridge claims.
Never create a gap for a fact already stated by the task or memory. A gap is
allowed only for a missing inferred bridge that can change the influence level,
and it must be a concrete proposition that evidence could support or refute.
Retrieval metadata is not evidence.

Express the complete influence bridge once in `bridge_steps`. Mark every step's
basis as task, memory, inference, or gap. A gap-based step must cite a gap_id.
`bridge_confidence` is confidence that the bridge as a whole holds, never the
size of its potential effect. An open bridge must contain at least one gap and a
gap-based bridge step. Supported/refuted bridges cannot request gaps. A none
potential impact uses a refuted bridge and no possible_action_change.

`source_type` describes where to obtain missing evidence, never the semantic
topic. Use only: personal_memory_store for additional personal memories;
external_knowledge for stable public knowledge; environment_tool for current
environment, records, or tools; user for information that only the user can
clarify. Facts already supplied by the task or candidate are supported claims,
not retrieval sources and not gaps. Do not invent domain-specific source types.

Return one JSON object with key `hypotheses`. Each hypothesis must contain:
candidate_id, potential_impact (high|low|none), effect_target,
target_dimension_ids, mechanism, bridge_steps, conditions,
task_supported_claims, memory_supported_claims, inferred_claims, gaps,
possible_action_change, bridge_status (supported|open|refuted), and
bridge_confidence. Each gap must contain gap_id,
proposition, source_type, retrieval_query, decision_relevance, priority, status
(use `open`). Array fields must be JSON arrays. Confidence and priority fields
are JSON numbers from 0.0 to 1.0, never labels. Follow the JSON Schema exactly.
"""

EVIDENCE_RELATION_PROMPT = """Judge how one acquired evidence item relates to one
specific Evidence Gap. This is a semantic judgment after retrieval and before
final influence validation.

Evidence Gap:
{gap_json}

Acquired evidence:
{evidence_json}

Use only the supplied evidence content and provenance. Classify the relation as:
- support: the evidence directly supports the gap proposition;
- refute: the evidence directly contradicts or rules out the gap proposition;
- insufficient: the evidence is related but does not decide the proposition.

Do not infer support merely because retrieval succeeded. Do not answer the
overall user task and do not change the gap. Return the supplied evidence_id and
gap_id unchanged. `grounding_spans` must be short verbatim substrings of the
evidence content that justify the relation; it may be empty only when no exact
span is informative. Return one JSON object with key `assessment`. Follow the
JSON Schema below exactly.
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

Each acquired item includes an independently judged relation to its gap. Use the
evidence content and its grounded relation assessment; do not treat retrieval
score alone as support. Do not use unsupported model knowledge as evidence. The
original candidate may be cited as `memory:<candidate_id>`; acquired evidence
must be cited by its supplied evidence_id.
Return one JSON object with key `result`, containing candidate_id, status
(validated|rejected|irrelevant|unresolved), potential_impact (copied unchanged
from the hypothesis), validated_impact (high|low|none|null), conclusion,
evidence_ids, unresolved_gap_ids, action_implication and validation_confidence.
A validated result uses high or low validated_impact and cites original memory
or acquired evidence. A rejected/irrelevant result uses none. An unresolved
result uses null, preserves the hypothesis potential_impact, and lists open gap
IDs. For an open bridge, validation must cite at least one acquired item judged
support; rejection must cite at least one acquired item judged refute. Evidence
absence must never silently downgrade potential impact. Follow the JSON Schema
below exactly.
"""
