"""Command-line entry point for preflight and incremental retrieval runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .datasets import default_data_paths, load_experiment_data, load_frozen_experiment_data
from .embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from .preflight import build_preflight_report
from .profiles import DEFAULT_PROFILES, get_profiles
from .runner import run_profiles, write_run_artifacts


def _add_data_paths(parser: argparse.ArgumentParser) -> None:
    defaults = default_data_paths()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="isolation-partitioned My-Mem-test/dataset root; overrides legacy paths",
    )
    for name, default in defaults.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)


def _load(args: argparse.Namespace):
    if args.dataset_root:
        return load_frozen_experiment_data(args.dataset_root)
    return load_experiment_data(
        test_cases_path=args.test_cases_path,
        memory_catalog_path=args.memory_catalog_path,
        representations_path=args.representations_path,
        source_bindings_path=args.source_bindings_path,
        query_requirements_path=args.query_requirements_path,
        registry_path=args.registry_path,
        memory_auxiliary_path=args.memory_auxiliary_path,
        query_auxiliary_path=args.query_auxiliary_path,
    )


def _profile_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    get_profiles(names)
    return names


def _preflight(args: argparse.Namespace) -> int:
    data = _load(args)
    profiles = get_profiles(_profile_names(args.profiles))
    report = build_preflight_report(data, profiles)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    summary = report["summary"] | report["auxiliary_coverage"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        print(f"full report: {args.output}")
    return 0


def _run(args: argparse.Namespace) -> int:
    data = _load(args)
    names = _profile_names(args.profiles)
    if args.embedding_backend == "hashing":
        embedder = HashingEmbedder(dimensions=args.hash_dimensions)
    else:
        embedder = OpenAICompatibleEmbedder.from_environment()
    results, metadata = run_profiles(
        data,
        names,
        embedder=embedder,
        allow_incomplete=args.allow_incomplete,
        union_depth=args.union_depth,
        view_budget=None if args.view_budget == 0 else args.view_budget,
    )
    output = args.output_dir
    if output is None:
        project_root = Path(__file__).resolve().parents[1]
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = project_root / "results" / "relation_vector_lab" / run_id
    paths = write_run_artifacts(
        output,
        data=data,
        results_by_profile=results,
        run_metadata=metadata,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="check data/profile readiness")
    _add_data_paths(preflight)
    preflight.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILES),
        help="comma-separated profile names",
    )
    preflight.add_argument("--output", type=Path)
    preflight.set_defaults(func=_preflight)

    run = subparsers.add_parser("run", help="run one or more experiment profiles")
    _add_data_paths(run)
    run.add_argument("--profiles", default="p0_keyword,p0_vector")
    run.add_argument("--embedding-backend", choices=("hashing", "openai"), default="hashing")
    run.add_argument("--hash-dimensions", type=int, default=256)
    run.add_argument("--allow-incomplete", action="store_true")
    run.add_argument("--union-depth", type=int, default=5)
    run.add_argument(
        "--view-budget",
        type=int,
        default=1,
        help="maximum views/bindings per side; 0 uses every available view",
    )
    run.add_argument("--output-dir", type=Path)
    run.set_defaults(func=_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

