"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze
from .benchmark import build_evaluation_set, download_sources
from .runner import CONDITIONS, run_experiment
from .scoring import apply_adjudications, rescore_file
from .validation import validate_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Download and adapt the pinned BIPIA subset")
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="Run local MLX inference")
    run.add_argument("--model", required=True, help="Local model snapshot directory")
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    run.add_argument("--resume", action="store_true")

    report = subparsers.add_parser("analyze", help="Calculate metrics and render charts")
    report.add_argument("raw_outputs", type=Path)
    report.add_argument("--output-dir", type=Path)

    validate = subparsers.add_parser("validate", help="Validate a completed run")
    validate.add_argument("run_dir", type=Path)

    rescore = subparsers.add_parser("rescore", help="Rescore preserved outputs without rerunning inference")
    rescore.add_argument("source", type=Path)
    rescore.add_argument("destination", type=Path)

    adjudicate = subparsers.add_parser("adjudicate", help="Apply recorded reviews to unresolved scores")
    adjudicate.add_argument("source", type=Path)
    adjudicate.add_argument("adjudications", type=Path)
    adjudicate.add_argument("destination", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        download_sources()
        print(build_evaluation_set(force=args.force))
    elif args.command == "run":
        output = run_experiment(
            model_path=args.model,
            output_dir=args.output_dir,
            limit=args.limit,
            conditions=tuple(args.conditions),
            resume=args.resume,
        )
        print(output)
    elif args.command == "analyze":
        print(json.dumps(analyze(args.raw_outputs, args.output_dir), indent=2))
    elif args.command == "validate":
        for message in validate_results(args.run_dir):
            print(message)
    elif args.command == "rescore":
        rescore_file(args.source, args.destination)
        print(args.destination)
    elif args.command == "adjudicate":
        apply_adjudications(args.source, args.adjudications, args.destination)
        print(args.destination)


if __name__ == "__main__":
    main()
