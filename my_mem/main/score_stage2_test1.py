"""Score one stage2_test1 run against the provisional pilot Gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from my_mem.evaluation import score_stage2_test1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", choices=("E0", "Eplus", "Eminus", "Enoise"))
    args = parser.parse_args()

    run_doc = json.loads(Path(args.run).read_text(encoding="utf-8"))
    gold_doc = json.loads(
        (Path(args.dataset_dir) / "gold.json").read_text(encoding="utf-8")
    )
    report = score_stage2_test1(gold_doc, run_doc, condition=args.condition)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(output), "key_metrics": report["key_metrics"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
