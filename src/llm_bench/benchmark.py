"""Core benchmarking engine."""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_bench.model import Model
from llm_bench.suites import load_suite
from llm_bench.runner import run_prompt


@dataclass
class BenchmarkResult:
    """Results from a complete benchmark run."""

    models: list[ModelScore] = field(default_factory=list)
    suite: str = ""
    total_prompts: int = 0
    timestamp: str = ""

    def summary(self) -> None:
        """Print formatted summary table."""
        header = f"{'Model':<20} {'Latency':<10} {'Cost/1K':<10} {'Hallu %':<10} {'Quality':<10} {'Errors':<8}"
        print(header)
        print("-" * len(header))
        for m in sorted(self.models, key=lambda x: x.quality_score, reverse=True):
            print(
                f"{m.model_name:<20} "
                f"{m.avg_latency:<10.2f} "
                f"${m.cost_per_1k:<9.4f} "
                f"{m.hallucination_rate:<10.1%} "
                f"{m.quality_score:<10.1f} "
                f"{m.error_rate:<8.1%}"
            )

    def export(self, path: str) -> None:
        """Export results to JSON."""
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)

    def report(self, path: str) -> None:
        """Generate markdown report."""
        md = self._generate_markdown()
        with open(path, "w") as f:
            f.write(md)

    def _to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "total_prompts": self.total_prompts,
            "timestamp": self.timestamp,
            "models": [
                {
                    "name": m.model_name,
                    "avg_latency": m.avg_latency,
                    "cost_per_1k": m.cost_per_1k,
                    "hallucination_rate": m.hallucination_rate,
                    "quality_score": m.quality_score,
                    "error_rate": m.error_rate,
                }
                for m in self.models
            ],
        }

    def _generate_markdown(self) -> str:
        lines = [
            f"# LLM Benchmark Report — {self.suite}",
            f"",
            f"**Prompts tested:** {self.total_prompts}  ",
            f"**Timestamp:** {self.timestamp}",
            f"",
            "| Model | Latency (s) | Cost/1K | Hallucination | Quality | Errors |",
            "|-------|-------------|---------|---------------|---------|--------|",
        ]
        for m in self.models:
            lines.append(
                f"| {m.model_name} | {m.avg_latency:.2f} | ${m.cost_per_1k:.4f} | "
                f"{m.hallucination_rate:.1%} | {m.quality_score:.1f}/10 | {m.error_rate:.1%} |"
            )
        return "\n".join(lines)


@dataclass
class ModelScore:
    """Aggregated scores for a single model."""

    model_name: str
    avg_latency: float = 0.0
    cost_per_1k: float = 0.0
    hallucination_rate: float = 0.0
    quality_score: float = 0.0
    error_rate: float = 0.0
    total_tokens: int = 0


class Benchmark:
    """Main benchmarking orchestrator."""

    def __init__(
        self,
        models: list[Model],
        suite: str = "general",
        runs_per_prompt: int = 3,
        parallel: bool = True,
        timeout: int = 30,
    ):
        self.models = models
        self.suite = suite
        self.runs_per_prompt = runs_per_prompt
        self.parallel = parallel
        self.timeout = timeout

    def run(self) -> BenchmarkResult:
        """Execute the full benchmark suite across all models."""
        prompts = load_suite(self.suite)
        result = BenchmarkResult(
            suite=self.suite,
            total_prompts=len(prompts),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        )

        for model in self.models:
            score = self._bench_model(model, prompts)
            result.models.append(score)

        return result

    def _bench_model(self, model: Model, prompts: list[dict]) -> ModelScore:
        """Benchmark a single model across all prompts."""
        latencies: list[float] = []
        costs: list[float] = []
        errors = 0
        total_runs = len(prompts) * self.runs_per_prompt

        for prompt in prompts:
            for _ in range(self.runs_per_prompt):
                try:
                    resp = run_prompt(model, prompt, timeout=self.timeout)
                    latencies.append(resp["latency"])
                    costs.append(resp["cost"])
                except Exception:
                    errors += 1

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        total_cost = sum(costs)
        total_tokens = sum(c * 1000 / 0.01 for c in costs) if costs else 1  # rough

        return ModelScore(
            model_name=model.name,
            avg_latency=avg_latency,
            cost_per_1k=total_cost / (total_tokens / 1000) if total_tokens else 0,
            error_rate=errors / total_runs if total_runs else 0,
        )
