"""llm-bench - Automated LLM provider benchmarking."""

__version__ = "0.1.0"

from llm_bench.benchmark import Benchmark
from llm_bench.model import Model
from llm_bench.hallucination import HallucinationChecker

__all__ = ["Benchmark", "Model", "HallucinationChecker"]
