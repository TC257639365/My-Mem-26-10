"""Deterministic evidence retrieval for the first controlled experiment."""

from __future__ import annotations

import re

from my_mem.schemas import AcquiredEvidence, EvidenceDocument, EvidenceGap


def _terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    words.update(cjk)
    words.update(cjk[i : i + 2] for i in range(max(0, len(cjk) - 1)))
    return {term for term in words if term}


class FixedCorpusRetriever:
    """Rank a supplied evidence pool by lexical overlap.

    This intentionally modest retriever keeps the first experiment focused on
    gap quality.  An embedding or tool-backed retriever can replace it later via
    the same interface.
    """

    async def retrieve(
        self,
        gap: EvidenceGap,
        documents: list[EvidenceDocument],
        *,
        limit: int,
    ) -> list[AcquiredEvidence]:
        query_terms = _terms(f"{gap.proposition} {gap.retrieval_query}")
        ranked: list[tuple[float, EvidenceDocument]] = []
        for document in documents:
            if gap.source_type not in ("any", "unknown") and document.source_type != gap.source_type:
                continue
            doc_terms = _terms(document.content)
            score = len(query_terms & doc_terms) / max(1, len(query_terms))
            if score > 0:
                ranked.append((score, document))
        ranked.sort(key=lambda row: (-row[0], row[1].document_id))
        return [
            AcquiredEvidence(
                evidence_id=f"{gap.gap_id}:{doc.document_id}",
                gap_id=gap.gap_id,
                document_id=doc.document_id,
                content=doc.content,
                source_type=doc.source_type,
                provenance=doc.provenance,
                retrieval_score=score,
            )
            for score, doc in ranked[:limit]
        ]

