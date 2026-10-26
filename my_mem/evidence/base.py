"""证据获取接口。

未来的个人记忆、外部知识、环境工具和用户询问都应实现同一协议，
这样替换证据来源时不需要修改 Controller。
"""

from __future__ import annotations

from typing import Protocol

from my_mem.schemas import AcquiredEvidence, EvidenceDocument, EvidenceGap


class EvidenceRetriever(Protocol):
    """输入一个 EvidenceGap，返回带来源信息的证据列表。"""

    async def retrieve(
        self,
        gap: EvidenceGap,
        documents: list[EvidenceDocument],
        *,
        limit: int,
        candidate_id: str | None = None,
    ) -> list[AcquiredEvidence]: ...


class EvidenceRelationJudge(Protocol):
    """判断已取得证据对具体 Gap 是支持、反驳还是不足。"""

    async def judge(
        self,
        gap: EvidenceGap,
        evidence: AcquiredEvidence,
    ) -> AcquiredEvidence: ...
