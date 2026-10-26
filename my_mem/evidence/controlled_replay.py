"""Stage2 受控实验的证据回放器。"""

from __future__ import annotations

from my_mem.schemas import AcquiredEvidence, EvidenceDocument, EvidenceGap


class ControlledEvidenceRetriever:
    """按 candidate 和 acquisition channel 回放预置证据。

    `EvidenceDocument.metadata` 中的 `target_candidate_id` 只用于实验路由，
    不复制到 AcquiredEvidence，因此不会把 Gold 角色或 condition 暴露给 LLM。
    该组件有意绕过词项排序，用于把“证据是否被取回”固定下来。
    """

    async def retrieve(
        self,
        gap: EvidenceGap,
        documents: list[EvidenceDocument],
        *,
        limit: int,
        candidate_id: str | None = None,
    ) -> list[AcquiredEvidence]:
        if not candidate_id:
            raise ValueError("controlled replay requires candidate_id")
        if limit <= 0:
            return []

        matched = [
            document
            for document in documents
            if document.source_type == gap.source_type
            and document.metadata.get("target_candidate_id") == candidate_id
        ]
        matched.sort(key=lambda document: document.document_id)
        return [
            AcquiredEvidence(
                evidence_id=f"{gap.gap_id}:{document.document_id}",
                gap_id=gap.gap_id,
                document_id=document.document_id,
                content=document.content,
                source_type=document.source_type,
                provenance=document.provenance,
                retrieval_score=1.0,
            )
            for document in matched[:limit]
        ]
