"""Future execution-policy boundary.

The minimal experiment stops at validated influences.  A later policy may map
task state and validated influences to RetrieveMemory, RetrieveKnowledge,
AskUser, RevisePlan, Execute or Stop.  Keeping this module empty prevents an
unvalidated policy layer from obscuring the first research question.
"""

from __future__ import annotations

from typing import Protocol

from my_mem.schemas import InfluenceRunResult


class ExecutionPolicy(Protocol):
    async def decide(self, state: InfluenceRunResult) -> str: ...


class NotImplementedExecutionPolicy:
    async def decide(self, state: InfluenceRunResult) -> str:
        raise NotImplementedError("Execution policy is intentionally deferred")

