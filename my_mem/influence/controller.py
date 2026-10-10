"""最小闭环的一轮式、确定性预算控制器。"""

from __future__ import annotations

from dataclasses import dataclass

from my_mem.evidence.base import EvidenceRetriever
from my_mem.influence.base import (
    HypothesisBuilder,
    InfluenceValidator,
    TaskFrameBuilder,
)
from my_mem.schemas import (
    AcquiredEvidence,
    InfluenceLevel,
    InfluenceRunResult,
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
    high_potential_impact: float = 0.60
    low_bridge_confidence: float = 0.60


class InfluenceController:
    def __init__(
        self,
        builder: HypothesisBuilder,
        retriever: EvidenceRetriever,
        validator: InfluenceValidator,
        config: ControllerConfig | None = None,
        *,
        task_frame_builder: TaskFrameBuilder,
    ):
        self.builder = builder
        self.retriever = retriever
        self.validator = validator
        self.task_frame_builder = task_frame_builder
        self.config = config or ControllerConfig()

    async def run(self, case: TaskCase) -> InfluenceRunResult:
        # 1. Candidate Discovery 已由 T-Mem 完成；这里只截取固定数量进入影响分析。
        candidates = case.candidates[: self.config.max_candidates]
        candidates_by_id = {candidate.memory_id: candidate for candidate in candidates}

        # 2. 仅根据 Query 生成公共 TaskFrame，再让所有候选使用同一基准。
        task_frame = await self.task_frame_builder.build(case.task)

        # 3. 批量构造 InfluenceHypothesis。候选最多在自身 hypothesis 中保存一个
        # 临时新维度；不会修改 task_frame，也不会触发二次候选召回。
        task_frame_snapshot = task_frame.model_dump(mode="json")
        hypotheses = await self.builder.build(case.task, task_frame, candidates)
        if task_frame.model_dump(mode="json") != task_frame_snapshot:
            raise ValueError("hypothesis builder must not mutate the common TaskFrame")

        # 4. 验证模块间 ID 和维度引用，防止绑定到错误记忆或虚构公共维度。
        hypothesis_ids = [hypothesis.candidate_id for hypothesis in hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)) or set(
            hypothesis_ids
        ) != set(candidates_by_id):
            raise ValueError("builder must return exactly one hypothesis per candidate")
        gap_ids = [gap.gap_id for hypothesis in hypotheses for gap in hypothesis.gaps]
        if len(gap_ids) != len(set(gap_ids)):
            raise ValueError("gap_id values must be unique within a case")
        common_dimension_ids = {
            dimension.dimension_id for dimension in task_frame.common_dimensions
        }
        for hypothesis in hypotheses:
            unknown_ids = set(hypothesis.matched_dimension_ids) - common_dimension_ids
            if unknown_ids:
                raise ValueError(
                    f"hypothesis cites unknown common dimension ids: {unknown_ids}"
                )
            if (
                hypothesis.proposed_dimension is not None
                and hypothesis.proposed_dimension.dimension_id in common_dimension_ids
            ):
                raise ValueError("proposed dimension id collides with a common dimension")

        # 5. 确定性调度：潜在影响大、桥接置信度低、gap 重要的命题优先。
        ranked_gaps = sorted(
            (
                (
                    hypothesis.impact
                    * (1.0 - hypothesis.bridge_confidence)
                    * gap.priority,
                    gap,
                )
                for hypothesis in hypotheses
                for gap in hypothesis.gaps
            ),
            key=lambda row: (-row[0], row[1].gap_id),
        )[: self.config.max_total_gaps]
        selected_gap_ids = {gap.gap_id for _, gap in ranked_gaps}

        # 6. 对选中的 gap 获取证据，并按 candidate_id 汇总给验证器。
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

        # 7. none 候选直接终止；high/low 候选由 Validator 综合原始记忆和新证据判断。
        validated: list[ValidatedInfluence] = []
        for hypothesis in hypotheses:
            if hypothesis.influence_level == InfluenceLevel.NONE:
                validated.append(
                    ValidatedInfluence(
                        candidate_id=hypothesis.candidate_id,
                        status=ValidationStatus.IRRELEVANT,
                        influence_level=InfluenceLevel.NONE,
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

        # 8. 标记“潜在影响高但桥接置信度低”的疑难样本。本版只记录，不追加
        # LLM 展开调用，也不进行二次候选召回。
        difficult_bridge_candidate_ids = [
            hypothesis.candidate_id
            for hypothesis in hypotheses
            if hypothesis.impact >= self.config.high_potential_impact
            and hypothesis.bridge_confidence < self.config.low_bridge_confidence
        ]

        # 9. 保存完整中间轨迹，而不只保存最终结论，方便做模块级指标和错误分析。
        all_evidence = [
            item
            for hypothesis in hypotheses
            for item in evidence_by_candidate[hypothesis.candidate_id]
        ]
        return InfluenceRunResult(
            case_id=case.case_id,
            task=case.task,
            task_frame=task_frame,
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
                "task_frame_revision_rounds": 0,
                "secondary_candidate_recall_rounds": 0,
                "bridge_reasoning_rounds": 1,
                "novel_dimension_count": sum(
                    h.proposed_dimension is not None for h in hypotheses
                ),
                "difficult_bridge_candidate_ids": difficult_bridge_candidate_ids,
            },
        )
