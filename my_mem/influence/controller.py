"""最小闭环的一轮式、确定性预算控制器。"""

from __future__ import annotations

from dataclasses import dataclass

from my_mem.evidence.base import EvidenceRetriever
from my_mem.influence.base import HypothesisBuilder, InfluenceValidator
from my_mem.schemas import (
    AcquiredEvidence,
    InfluenceRunResult,
    InfluenceStatus,
    TaskCase,
    ValidatedInfluence,
    ValidationStatus,
)


@dataclass(frozen=True)
class ControllerConfig:
    # 这些预算由代码强制执行，不交给 LLM 自行决定。
    max_candidates: int = 8
    max_total_gaps: int = 5
    max_evidence_per_gap: int = 3


class InfluenceController:
    def __init__(
        self,
        builder: HypothesisBuilder,
        retriever: EvidenceRetriever,
        validator: InfluenceValidator,
        config: ControllerConfig | None = None,
    ):
        self.builder = builder
        self.retriever = retriever
        self.validator = validator
        self.config = config or ControllerConfig()

    async def run(self, case: TaskCase) -> InfluenceRunResult:
        # 1. Candidate Discovery 已由 T-Mem 完成；这里只截取固定数量进入影响分析。
        candidates = case.candidates[: self.config.max_candidates]
        candidates_by_id = {candidate.memory_id: candidate for candidate in candidates}

        # 2. 批量构造 InfluenceHypothesis，其中可包含零个或多个 EvidenceGap。
        hypotheses = await self.builder.build(case.task, candidates)

        # 3. 验证模块间 ID 完整性，防止后续把假设或证据绑定到错误记忆。
        hypothesis_ids = [hypothesis.candidate_id for hypothesis in hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)) or set(hypothesis_ids) != set(candidates_by_id):
            raise ValueError("builder must return exactly one hypothesis per candidate")
        gap_ids = [gap.gap_id for hypothesis in hypotheses for gap in hypothesis.gaps]
        if len(gap_ids) != len(set(gap_ids)):
            raise ValueError("gap_id values must be unique within a case")

        # 4. 确定性调度：优先展开“影响大、当前不确定、gap 本身重要”的命题。
        ranked_gaps = sorted(
            (
                (hypothesis.impact * hypothesis.uncertainty * gap.priority, gap)
                for hypothesis in hypotheses
                for gap in hypothesis.gaps
            ),
            key=lambda row: (-row[0], row[1].gap_id),
        )[: self.config.max_total_gaps]
        selected_gap_ids = {gap.gap_id for _, gap in ranked_gaps}

        # 5. 对选中的 gap 获取证据，并按 candidate_id 汇总给验证器。
        evidence_by_candidate: dict[str, list[AcquiredEvidence]] = {
            hypothesis.candidate_id: [] for hypothesis in hypotheses
        }
        for hypothesis in hypotheses:
            for gap in hypothesis.gaps:
                if gap.gap_id not in selected_gap_ids:
                    continue
                acquired = await self.retriever.retrieve(
                    gap,
                    case.evidence_pool,
                    limit=self.config.max_evidence_per_gap,
                )
                evidence_by_candidate[hypothesis.candidate_id].extend(acquired)

        # 6. irrelevant 候选直接终止；其他候选由 Validator 综合原始记忆和新证据判断。
        validated: list[ValidatedInfluence] = []
        for hypothesis in hypotheses:
            if hypothesis.status == InfluenceStatus.IRRELEVANT:
                validated.append(
                    ValidatedInfluence(
                        candidate_id=hypothesis.candidate_id,
                        status=ValidationStatus.IRRELEVANT,
                        conclusion="Candidate was screened out before evidence acquisition.",
                        confidence=1.0,
                    )
                )
                continue
            validated.append(
                await self.validator.validate(
                    case.task,
                    candidates_by_id[hypothesis.candidate_id],
                    hypothesis,
                    evidence_by_candidate[hypothesis.candidate_id],
                )
            )

        # 7. 保存完整中间轨迹，而不只保存最终结论，方便做模块级指标和错误分析。
        all_evidence = [
            item
            for hypothesis in hypotheses
            for item in evidence_by_candidate[hypothesis.candidate_id]
        ]
        return InfluenceRunResult(
            case_id=case.case_id,
            task=case.task,
            hypotheses=hypotheses,
            acquired_evidence=all_evidence,
            validated_influences=validated,
            diagnostics={
                "candidate_count_input": len(case.candidates),
                "candidate_count_used": len(candidates),
                "gap_count_proposed": sum(len(h.gaps) for h in hypotheses),
                "gap_count_selected": len(selected_gap_ids),
                "evidence_count": len(all_evidence),
                "acquisition_rounds": 1,
            },
        )
