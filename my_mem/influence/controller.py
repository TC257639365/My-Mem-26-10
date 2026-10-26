"""最小闭环的一轮式、确定性预算控制器。"""

from __future__ import annotations

from dataclasses import dataclass

from my_mem.evidence.base import EvidenceRelationJudge, EvidenceRetriever
from my_mem.influence.base import (
    DimensionChallengeBuilder,
    HypothesisBuilder,
    InfluenceValidator,
    TaskFrameBuilder,
)
from my_mem.schemas import (
    AcquiredEvidence,
    BridgeStatus,
    EvidenceRelation,
    ImpactLevel,
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
        dimension_challenge_builder: DimensionChallengeBuilder,
        evidence_relation_judge: EvidenceRelationJudge | None = None,
    ):
        self.builder = builder
        self.retriever = retriever
        self.validator = validator
        self.task_frame_builder = task_frame_builder
        self.dimension_challenge_builder = dimension_challenge_builder
        self.evidence_relation_judge = evidence_relation_judge
        self.config = config or ControllerConfig()

    async def run(self, case: TaskCase) -> InfluenceRunResult:
        # 1. Candidate Discovery 已由 T-Mem 完成；这里只截取固定数量进入影响分析。
        candidates = case.candidates[: self.config.max_candidates]
        candidates_by_id = {candidate.memory_id: candidate for candidate in candidates}

        # 2. 仅根据 Query 生成公共 TaskFrame，再让所有候选使用同一基准。
        task_frame = await self.task_frame_builder.build(case.task)

        # 3. 在任何影响定级之前，让每个候选独立映射 F0 或提出一次局部维度挑战。
        # 不确定的身份/事件桥接不会在这里把候选提前压成 none。
        task_frame_snapshot = task_frame.model_dump(mode="json")
        dimension_challenges = await self.dimension_challenge_builder.build(
            case.task, task_frame, candidates
        )
        if task_frame.model_dump(mode="json") != task_frame_snapshot:
            raise ValueError("dimension challenge builder must not mutate TaskFrame")

        challenge_ids = [item.candidate_id for item in dimension_challenges]
        if len(challenge_ids) != len(set(challenge_ids)) or set(challenge_ids) != set(
            candidates_by_id
        ):
            raise ValueError("challenge builder must return exactly one item per candidate")
        common_dimension_ids = {
            dimension.dimension_id for dimension in task_frame.common_dimensions
        }
        challenges_by_id = {item.candidate_id: item for item in dimension_challenges}
        for challenge in dimension_challenges:
            unknown_ids = set(challenge.matched_dimension_ids) - common_dimension_ids
            if unknown_ids:
                raise ValueError(f"challenge cites unknown common dimensions: {unknown_ids}")
            if (
                challenge.proposed_dimension is not None
                and challenge.proposed_dimension.dimension_id in common_dimension_ids
            ):
                raise ValueError("proposed dimension id collides with a common dimension")

        # 4. 只有完成维度审查后才构造影响假设。potential_impact 表达桥接成立时的
        # 影响上界，bridge_status/confidence 单独表达桥接确定性。
        hypotheses = await self.builder.build(
            case.task, task_frame, candidates, dimension_challenges
        )
        if task_frame.model_dump(mode="json") != task_frame_snapshot:
            raise ValueError("hypothesis builder must not mutate the common TaskFrame")

        # 5. 验证模块间 ID、Gap 和审查后维度引用。
        hypothesis_ids = [hypothesis.candidate_id for hypothesis in hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)) or set(
            hypothesis_ids
        ) != set(candidates_by_id):
            raise ValueError("builder must return exactly one hypothesis per candidate")
        gap_ids = [gap.gap_id for hypothesis in hypotheses for gap in hypothesis.gaps]
        if len(gap_ids) != len(set(gap_ids)):
            raise ValueError("gap_id values must be unique within a case")
        for hypothesis in hypotheses:
            challenge = challenges_by_id[hypothesis.candidate_id]
            allowed_dimension_ids = set(challenge.matched_dimension_ids)
            if challenge.proposed_dimension is not None:
                allowed_dimension_ids.add(challenge.proposed_dimension.dimension_id)
            unknown_ids = set(hypothesis.target_dimension_ids) - allowed_dimension_ids
            if unknown_ids:
                raise ValueError(
                    f"hypothesis cites dimensions not approved by challenge: {unknown_ids}"
                )

        # 6. 确定性调度：使用“桥接成立时的影响上界”，不再用未确认桥接压低
        # 影响分。这样 high-potential/open 候选会优先得到补证据机会。
        impact_weights = {
            ImpactLevel.HIGH: 1.0,
            ImpactLevel.LOW: 0.5,
            ImpactLevel.NONE: 0.0,
        }
        ranked_gaps = sorted(
            (
                (
                    impact_weights[hypothesis.potential_impact]
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

        # 7. 对选中的 gap 获取证据，并按 candidate_id 汇总给验证器。
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
                    candidate_id=hypothesis.candidate_id,
                )
                if self.evidence_relation_judge is not None:
                    acquired = [
                        await self.evidence_relation_judge.judge(gap, item)
                        for item in acquired
                    ]
                evidence_by_candidate[hypothesis.candidate_id].extend(acquired)

        # 8. 无条件影响的候选可终止；refuted 桥接被拒绝；open 桥接在没有取得
        # 新证据时保持 unresolved，禁止 Validator 仅因证据缺失下调影响上界。
        validated: list[ValidatedInfluence] = []
        for hypothesis in hypotheses:
            if hypothesis.potential_impact == ImpactLevel.NONE:
                validated.append(
                    ValidatedInfluence(
                        candidate_id=hypothesis.candidate_id,
                        status=ValidationStatus.IRRELEVANT,
                        potential_impact=ImpactLevel.NONE,
                        validated_impact=ImpactLevel.NONE,
                        conclusion="No material response or action delta exists.",
                        validation_confidence=1.0,
                    )
                )
                continue
            if hypothesis.bridge_status == BridgeStatus.REFUTED:
                validated.append(
                    ValidatedInfluence(
                        candidate_id=hypothesis.candidate_id,
                        status=ValidationStatus.REJECTED,
                        potential_impact=hypothesis.potential_impact,
                        validated_impact=ImpactLevel.NONE,
                        conclusion="The required influence bridge is refuted.",
                        validation_confidence=1.0 - hypothesis.bridge_confidence,
                    )
                )
                continue
            candidate_evidence = evidence_by_candidate[hypothesis.candidate_id]
            if hypothesis.bridge_status == BridgeStatus.OPEN and not candidate_evidence:
                validated.append(
                    ValidatedInfluence(
                        candidate_id=hypothesis.candidate_id,
                        status=ValidationStatus.UNRESOLVED,
                        potential_impact=hypothesis.potential_impact,
                        validated_impact=None,
                        conclusion=(
                            "The potential impact is preserved because no new evidence "
                            "resolved the open bridge."
                        ),
                        unresolved_gap_ids=[gap.gap_id for gap in hypothesis.gaps],
                        action_implication=hypothesis.possible_action_change,
                        validation_confidence=0.0,
                    )
                )
                continue
            # 相关但不足的证据仍然没有解决 Gap。只有 relation judge 已实际运行
            # 且所有结果都是 insufficient 时才在此确定性终止；unjudged 保留旧组件
            # 的兼容行为，仍交给 Validator。
            if (
                hypothesis.bridge_status == BridgeStatus.OPEN
                and candidate_evidence
                and all(
                    item.relation == EvidenceRelation.INSUFFICIENT
                    for item in candidate_evidence
                )
            ):
                validated.append(
                    ValidatedInfluence(
                        candidate_id=hypothesis.candidate_id,
                        status=ValidationStatus.UNRESOLVED,
                        potential_impact=hypothesis.potential_impact,
                        validated_impact=None,
                        conclusion=(
                            "Retrieved evidence was insufficient to resolve the open bridge."
                        ),
                        evidence_ids=[item.evidence_id for item in candidate_evidence],
                        unresolved_gap_ids=[gap.gap_id for gap in hypothesis.gaps],
                        action_implication=hypothesis.possible_action_change,
                        validation_confidence=0.0,
                    )
                )
                continue
            validated.append(
                await self.validator.validate(
                    case.task,
                    candidates_by_id[hypothesis.candidate_id],
                    hypothesis,
                    candidate_evidence,
                )
            )

        # 9. 标记“潜在影响高但桥接置信度低”的疑难样本。本版只记录，不追加
        # LLM 展开调用，也不进行二次候选召回。
        difficult_bridge_candidate_ids = [
            hypothesis.candidate_id
            for hypothesis in hypotheses
            if hypothesis.potential_impact == ImpactLevel.HIGH
            and hypothesis.bridge_confidence < self.config.low_bridge_confidence
        ]

        # 10. 保存完整中间轨迹，而不只保存最终结论，方便做模块级指标和错误分析。
        all_evidence = [
            item
            for hypothesis in hypotheses
            for item in evidence_by_candidate[hypothesis.candidate_id]
        ]
        return InfluenceRunResult(
            case_id=case.case_id,
            task=case.task,
            task_frame=task_frame,
            dimension_challenges=dimension_challenges,
            hypotheses=hypotheses,
            acquired_evidence=all_evidence,
            validated_influences=validated,
            diagnostics={
                "candidate_count_input": len(case.candidates),
                "candidate_count_used": len(candidates),
                "gap_count_proposed": sum(len(h.gaps) for h in hypotheses),
                "gap_count_selected": len(selected_gap_ids),
                "evidence_count": len(all_evidence),
                "evidence_relation_counts": {
                    relation.value: sum(
                        item.relation == relation for item in all_evidence
                    )
                    for relation in EvidenceRelation
                },
                "acquisition_rounds": 1,
                "task_frame_revision_rounds": 0,
                "secondary_candidate_recall_rounds": 0,
                "bridge_reasoning_rounds": 1,
                "novel_dimension_count": sum(
                    item.proposed_dimension is not None
                    for item in dimension_challenges
                ),
                "difficult_bridge_candidate_ids": difficult_bridge_candidate_ids,
            },
        )
