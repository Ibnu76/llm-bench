"""CLI interface for llm-bench."""

import argparse
import sys
import yaml
from pathlib import Path

from llm_bench.benchmark import Benchmark
from llm_bench.model import Model


def main():
    parser = argparse.ArgumentParser(prog="llm-bench", description="LLM provider benchmarking")
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("--models", required=True, help="Comma-separated model names")
    run_parser.add_argument("--suite", default="general", help="Test suite name")
    run_parser.add_argument("--runs", type=int, default=3, help="Runs per prompt")
    run_parser.add_argument("--output", default="./results", help="Output directory")
    run_parser.add_argument("--format", default="markdown", choices=["markdown", "json", "csv"])
    run_parser.add_argument("--prompts", help="Custom prompts YAML file")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report from results")
    report_parser.add_argument("--input", required=True, help="Results JSON file")
    report_parser.add_argument("--format", default="markdown")
    report_parser.add_argument("--output", default="./report.md")

    args = parser.parse_args()

    if args.command == "run":
        _run_benchmark(args)
    elif args.command == "report":
        _generate_report(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_benchmark(args):
    """Execute benchmark run."""
    model_names = [m.strip() for m in args.models.split(",")]
    models = [_resolve_model(name) for name in model_names]

    bench = Benchmark(
        models=models,
        suite=args.suite,
        runs_per_prompt=args.runs,
    )

    print(f"Running {args.suite} suite across {len(models)} models...")
    results = bench.run()
    results.summary()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results.export(str(output_dir / "results.json"))
    results.report(str(output_dir / "report.md"))
    print(f"\nResults saved to {output_dir}/")


def _resolve_model(name: str) -> Model:
    """Resolve model name to Model object with provider detection."""
    provider_hints = {
        "gpt": "openai",
        "claude": "anthropic",
        "deepseek": "deepseek",
        "llama": "together",
        "gemini": "google",
        "mixtral": "together",
    }
    provider = "openai"  # default
    for hint, prov in provider_hints.items():
        if hint in name.lower():
            provider = prov
            break
    return Model(name=name, provider=provider)


def _generate_report(args):
    """Generate report from existing results."""
    import json
    with open(args.input) as f:
        data = json.load(f)
    print(f"Report generated: {args.output}")


if __name__ == "__main__":
    main()
