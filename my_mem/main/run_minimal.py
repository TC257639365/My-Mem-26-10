"""Run the one-round My-Mem prototype over standalone TaskCase JSON."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from T_mem.llm.llm_provider import LLMProvider
from my_mem.adapters.tmem import load_case_file
from my_mem.evidence.fixed_corpus import FixedCorpusRetriever
from my_mem.influence.controller import ControllerConfig, InfluenceController
from my_mem.influence.llm_components import LLMHypothesisBuilder, LLMInfluenceValidator


async def _run(args: argparse.Namespace) -> None:
    cases = load_case_file(args.input)
    provider = LLMProvider(model=args.model)
    controller = InfluenceController(
        builder=LLMHypothesisBuilder(provider),
        retriever=FixedCorpusRetriever(),
        validator=LLMInfluenceValidator(provider),
        config=ControllerConfig(
            max_candidates=args.max_candidates,
            max_total_gaps=args.max_gaps,
            max_evidence_per_gap=args.max_evidence_per_gap,
        ),
    )
    results = []
    for case in cases:
        results.append((await controller.run(case)).model_dump(mode="json"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-gaps", type=int, default=5)
    parser.add_argument("--max-evidence-per-gap", type=int, default=3)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()

