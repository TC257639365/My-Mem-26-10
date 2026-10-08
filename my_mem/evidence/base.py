"""Evidence acquisition interface.

Future providers (personal memory, web knowledge, environment tools and user
questions) should implement this contract without changing the controller.
"""

from __future__ import annotations

from typing import Protocol

from my_mem.schemas import AcquiredEvidence, EvidenceDocument, EvidenceGap


class EvidenceRetriever(Protocol):
    async def retrieve(
        self,
        gap: EvidenceGap,
        documents: list[EvidenceDocument],
        *,
        limit: int,
    ) -> list[AcquiredEvidence]: ...

