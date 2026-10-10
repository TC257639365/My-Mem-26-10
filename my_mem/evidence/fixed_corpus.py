"""首轮受控实验使用的确定性证据检索器。"""

from __future__ import annotations

import re

from my_mem.schemas import (
    AcquiredEvidence,
    EvidenceDocument,
    EvidenceGap,
)


def _terms(text: str) -> set[str]:
    # 英文按词切分；中文同时保留单字和二元片段，避免额外分词依赖。
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    words.update(cjk)
    words.update(cjk[i : i + 2] for i in range(max(0, len(cjk) - 1)))
    return {term for term in words if term}


class FixedCorpusRetriever:
    """按照词项重合率排序固定证据池。

    这里故意不用复杂 Retriever，使第一轮实验主要测量 Evidence Gap 的质量；
    后续 embedding 或工具检索器可以通过相同接口替换它。
    """

    async def retrieve(
        self,
        gap: EvidenceGap,
        documents: list[EvidenceDocument],
        *,
        limit: int,
    ) -> list[AcquiredEvidence]:
        # proposition 表达“要验证什么”，retrieval_query 表达“应该搜什么”。
        query_terms = _terms(f"{gap.proposition} {gap.retrieval_query}")
        ranked: list[tuple[float, EvidenceDocument]] = []
        for document in documents:
            # 先按证据源路由，避免知识问题误用环境记录作答。
            if document.source_type != gap.source_type:
                continue
            doc_terms = _terms(document.content)
            score = len(query_terms & doc_terms) / max(1, len(query_terms))
            if score > 0:
                ranked.append((score, document))
        ranked.sort(key=lambda row: (-row[0], row[1].document_id))
        # AcquiredEvidence 保存 gap_id 和 provenance，使验证器只能引用真实取回的证据。
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
