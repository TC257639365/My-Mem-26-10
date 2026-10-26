"""运行 My-Mem 一轮最小闭环。

输入既可以是手工构造的 TaskCase，也可以是改造后 T-Mem Stage 6 生成的
search_results.json。后者需要通过 --evidence-pool 单独提供受控证据池。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from T_mem.llm.llm_provider import LLMProvider
from my_mem.adapters.tmem import load_case_file, load_cases, load_evidence_pool
from my_mem.evidence.fixed_corpus import FixedCorpusRetriever
from my_mem.influence.controller import ControllerConfig, InfluenceController
from my_mem.influence.llm_components import (
    LLMDimensionChallengeBuilder,
    LLMEvidenceRelationJudge,
    LLMHypothesisBuilder,
    LLMInfluenceValidator,
    LLMTaskFrameBuilder,
)


async def _run(args: argparse.Namespace) -> None:
    # T-Mem 输出只携带候选记忆；证据池独立注入，便于固定候选做公平消融。
    if args.input_format == "tmem":
        pool = load_evidence_pool(args.evidence_pool) if args.evidence_pool else []
        cases = load_cases(args.input, evidence_pool=pool)
    else:
        cases = load_case_file(args.input)

    # 复用 T-Mem 的 OpenAI-compatible Provider，同时复用其重试和成本日志能力。
    # --model 仅保留为临时覆盖；正常运行从 cloud.env/local.env 的
    # T_MEM_LLM_BASE_MODEL 统一读取模型名。
    provider = LLMProvider(model=args.model)

    # 语义判断由 LLM 完成；预算、排序和停止条件由确定性 Controller 完成。
    controller = InfluenceController(
        builder=LLMHypothesisBuilder(provider),
        retriever=FixedCorpusRetriever(),
        validator=LLMInfluenceValidator(provider),
        task_frame_builder=LLMTaskFrameBuilder(provider),
        dimension_challenge_builder=LLMDimensionChallengeBuilder(provider),
        evidence_relation_judge=LLMEvidenceRelationJudge(provider),
        config=ControllerConfig(
            max_candidates=args.max_candidates,
            max_total_gaps=args.max_gaps,
            max_evidence_per_gap=args.max_evidence_per_gap,
        ),
    )
    # 当前按 case 顺序运行，先保证结果可复现；并发和逐 case 恢复留待后续完善。
    results = []
    for case in cases:
        results.append((await controller.run(case)).model_dump(mode="json"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--input-format",
        choices=("case", "tmem"),
        default="case",
        help="case=TaskCase JSON；tmem=T-Mem Stage 6 search_results.json",
    )
    parser.add_argument(
        "--evidence-pool",
        help="input-format=tmem 时可选的 EvidenceDocument JSON 文件",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        help="临时覆盖 T_MEM_LLM_BASE_MODEL；通常无需传入",
    )
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-gaps", type=int, default=5)
    parser.add_argument("--max-evidence-per-gap", type=int, default=3)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
