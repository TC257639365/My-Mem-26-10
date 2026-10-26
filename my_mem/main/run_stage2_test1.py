"""Run the fixed non-ablation stage2_test1 configuration on the pilot dataset."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from T_mem.llm.llm_provider import LLMProvider
from my_mem.adapters.stage2_pilot import (
    EVIDENCE_CONDITIONS,
    dataset_fingerprint,
    load_stage2_gold,
    load_stage2_test1_cases,
)
from my_mem.evaluation import score_stage2_test1
from my_mem.evidence.controlled_replay import ControlledEvidenceRetriever
from my_mem.influence.controller import ControllerConfig, InfluenceController
from my_mem.influence.llm_components import (
    LLMDimensionChallengeBuilder,
    LLMEvidenceRelationJudge,
    LLMHypothesisBuilder,
    LLMInfluenceValidator,
    LLMTaskFrameBuilder,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class RecordingProvider:
    """Record raw completions and prompt hashes without changing provider behavior."""

    def __init__(self, provider: LLMProvider, trace_path: Path, *, record_prompts: bool):
        self.provider = provider
        self.model = provider.model
        self.trace_path = trace_path
        self.record_prompts = record_prompts
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        response = await self.provider.generate(prompt, **kwargs)
        record = {
            "timestamp": _utc_now(),
            "call_site": kwargs.get("call_site"),
            "model": self.model,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "response": response,
        }
        if self.record_prompts:
            record["prompt"] = prompt
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return response


def _new_run_document(args: argparse.Namespace, provider: RecordingProvider) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).resolve()
    code_root = Path(__file__).resolve().parents[1]
    return {
        "experiment_id": "stage2_test1",
        "run_schema_version": "stage2-test1-run-v1",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "config": {
            "pipeline": (
                "automatic F0 + Dimension Challenge + Hypothesis + controlled "
                "evidence + relation judge + validation"
            ),
            "evidence_condition": args.condition,
            "model": provider.model,
            "temperature": 0.0,
            "max_candidates": args.max_candidates,
            "max_total_gaps": args.max_gaps,
            "max_evidence_per_gap": args.max_evidence_per_gap,
            "dataset_dir": str(dataset_dir),
            "dataset_fingerprint": dataset_fingerprint(dataset_dir),
            "code_fingerprint": {
                name: _sha256(code_root / relative_path)
                for name, relative_path in {
                    "prompts.py": Path("influence/prompts.py"),
                    "controller.py": Path("influence/controller.py"),
                    "llm_components.py": Path("influence/llm_components.py"),
                    "controlled_replay.py": Path("evidence/controlled_replay.py"),
                    "stage2_pilot_adapter.py": Path("adapters/stage2_pilot.py"),
                    "schemas.py": Path("schemas.py"),
                }.items()
            },
            "candidate_mode": "six-candidate batch",
            "retriever": "ControlledEvidenceRetriever",
            "relation_judge": "LLMEvidenceRelationJudge",
            "frame_ablation": "none",
            "case_limit": args.limit,
            "record_prompts": args.record_prompts,
            "trace_output": str(
                Path(args.trace_output or f"{args.output}.trace.jsonl").resolve()
            ),
        },
        "results": [],
        "errors": [],
    }


def _load_or_initialize(
    args: argparse.Namespace, provider: RecordingProvider, output: Path
) -> dict[str, Any]:
    if not output.exists() or not args.resume:
        return _new_run_document(args, provider)
    payload = json.loads(output.read_text(encoding="utf-8"))
    expected = _new_run_document(args, provider)["config"]
    for key in (
        "evidence_condition",
        "model",
        "max_candidates",
        "max_total_gaps",
        "max_evidence_per_gap",
        "dataset_fingerprint",
        "code_fingerprint",
        "case_limit",
        "record_prompts",
        "trace_output",
    ):
        if payload["config"].get(key) != expected.get(key):
            raise ValueError(f"cannot resume: config mismatch for {key}")
    return payload


async def _run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    trace_path = Path(args.trace_output or f"{args.output}.trace.jsonl")
    if trace_path.exists() and not args.resume:
        trace_path.unlink()
    base_provider = LLMProvider(model=args.model, temperature=0.0)
    provider = RecordingProvider(
        base_provider,
        trace_path,
        record_prompts=args.record_prompts,
    )
    payload = _load_or_initialize(args, provider, output)
    cases = load_stage2_test1_cases(args.dataset_dir, condition=args.condition)
    if args.limit is not None:
        cases = cases[: args.limit]

    controller = InfluenceController(
        builder=LLMHypothesisBuilder(provider),
        retriever=ControlledEvidenceRetriever(),
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

    completed = {item["case_id"] for item in payload["results"]}
    for case in cases:
        if case.case_id in completed:
            continue
        try:
            result = await controller.run(case)
            payload["results"].append(result.model_dump(mode="json"))
            payload["errors"] = [
                item for item in payload["errors"] if item["case_id"] != case.case_id
            ]
        except Exception as exc:  # noqa: BLE001 - case-level recovery is intentional
            payload["errors"] = [
                item for item in payload["errors"] if item["case_id"] != case.case_id
            ]
            payload["errors"].append(
                {
                    "case_id": case.case_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "timestamp": _utc_now(),
                }
            )
            if args.fail_fast:
                payload["updated_at"] = _utc_now()
                _atomic_write_json(output, payload)
                raise
        payload["updated_at"] = _utc_now()
        _atomic_write_json(output, payload)

    payload["completed_at"] = _utc_now()
    payload["updated_at"] = payload["completed_at"]
    _atomic_write_json(output, payload)

    if args.score_output:
        report = score_stage2_test1(
            load_stage2_gold(args.dataset_dir),
            payload,
            condition=args.condition,
        )
        _atomic_write_json(Path(args.score_output), report)
    print(
        json.dumps(
            {
                "experiment_id": payload["experiment_id"],
                "condition": args.condition,
                "completed": len(payload["results"]),
                "errors": len(payload["errors"]),
                "output": str(output),
                "trace": str(trace_path),
                "score_output": args.score_output,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-output")
    parser.add_argument("--trace-output")
    parser.add_argument("--condition", choices=EVIDENCE_CONDITIONS, default="Eplus")
    parser.add_argument("--model")
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-gaps", type=int, default=12)
    parser.add_argument("--max-evidence-per-gap", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--record-prompts", action="store_true")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
