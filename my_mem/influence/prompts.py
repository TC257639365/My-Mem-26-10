"""Prompts for the two semantic operations in the minimal loop."""

HYPOTHESIS_PROMPT = """You analyze whether retrieved memories can change a task decision.

Task:
{task}

Candidate memories:
{candidates_json}

For every candidate, return exactly one hypothesis. Separate claims stated by
the memory from inferences. A gap must be a concrete proposition that external
evidence could support or refute. Retrieval metadata is not evidence. Mark a
candidate irrelevant when it cannot plausibly change the decision.

Return one JSON object with key `hypotheses`. Each hypothesis must contain:
candidate_id, status (direct|conditional|irrelevant|unknown), effect_target,
mechanism, conditions, memory_supported_claims, inferred_claims, gaps,
possible_action_change, impact, uncertainty. Each gap must contain gap_id,
proposition, source_type, retrieval_query, decision_relevance, priority, status
(use `open`).
"""


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
(validated|rejected|irrelevant|unresolved), conclusion, evidence_ids,
unresolved_gap_ids, action_implication and confidence. A validated result must
cite at least one supplied evidence_id.
"""

