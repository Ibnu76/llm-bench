# llm-bench

Automated LLM benchmarking across providers. Test latency, cost, hallucination rate, and response quality — then generate comparison reports.

## Why

Choosing between LLM providers is guesswork without data. `llm-bench` runs standardized tests across models and gives you hard numbers:

- **Latency** — TTFB, tokens/sec, total response time
- **Cost** — actual $ per query, per 1K tokens, projected monthly
- **Hallucination rate** — factual accuracy on verifiable claims
- **Quality** — coherence, instruction-following, format compliance
- **Reliability** — error rates, timeout rates, consistency

## Install

```bash
pip install llm-bench
```

## Quick Start

```bash
# Benchmark 3 models on a standard test suite
llm-bench run --models gpt-4o,claude-sonnet-4-20250514,gemini-2.0-flash --suite general

# Custom prompts
llm-bench run --models gpt-4o,deepseek-v3 --prompts ./my-prompts.yaml

# Generate report
llm-bench report --format markdown --output results/
```

## Python API

```python
from llm_bench import Benchmark, Model

bench = Benchmark(
    models=[
        Model("gpt-4o", provider="openai"),
        Model("claude-sonnet-4-20250514", provider="anthropic"),
        Model("deepseek-chat", provider="deepseek"),
        Model("llama-3.1-70b", provider="together"),
    ],
    suite="general",      # or "coding", "reasoning", "factual"
    runs_per_prompt=3,    # statistical significance
    parallel=True,
)

results = bench.run()
results.summary()
results.export("benchmark_results.json")
results.report("report.md")
```

## Output Example

```
┌────────────────────┬──────────┬───────────┬──────────┬───────────┬──────────┐
│ Model              │ Latency  │ Cost/1K   │ Hallu %  │ Quality   │ Errors   │
├────────────────────┼──────────┼───────────┼──────────┼───────────┼──────────┤
│ gpt-4o             │ 1.2s     │ $0.0075   │ 4.2%     │ 8.7/10    │ 0.1%     │
│ claude-sonnet-4    │ 1.8s     │ $0.0090   │ 2.1%     │ 9.1/10    │ 0.0%     │
│ deepseek-chat      │ 2.1s     │ $0.0014   │ 6.8%     │ 8.2/10    │ 0.3%     │
│ llama-3.1-70b      │ 0.9s     │ $0.0009   │ 8.4%     │ 7.8/10    │ 0.5%     │
└────────────────────┴──────────┴───────────┴──────────┴───────────┴──────────┘

Winner by cost-efficiency: deepseek-chat (quality/$ ratio: 5857)
Winner by quality: claude-sonnet-4 (9.1/10, lowest hallucination)
Winner by speed: llama-3.1-70b (0.9s avg TTFB)
```

## Test Suites

| Suite | Tests | What it measures |
|-------|-------|-----------------|
| `general` | 50 prompts | Overall capability across domains |
| `coding` | 40 prompts | Code generation, debugging, review |
| `reasoning` | 30 prompts | Logic, math, multi-step problems |
| `factual` | 60 prompts | Verifiable claims, hallucination prone |
| `instruction` | 25 prompts | Format compliance, constraint following |
| `creative` | 20 prompts | Writing quality, originality |

## Hallucination Detection

```python
from llm_bench import HallucinationChecker

checker = HallucinationChecker(
    method="claim_verification",  # or "semantic_entropy", "self_consistency"
    verifier_model="gpt-4o",      # judge model
)

# Check a single response
score = checker.check(
    prompt="When was Python 3.12 released?",
    response="Python 3.12 was released on October 2, 2023.",
    ground_truth="October 2, 2023",  # optional
)
print(f"Hallucination score: {score.risk}%")
```

## Configuration

```yaml
# llm-bench.yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
  deepseek:
    api_key: ${DEEPSEEK_API_KEY}
    base_url: https://api.deepseek.com/v1

benchmark:
  runs_per_prompt: 3
  timeout: 30
  parallel_requests: 5
  
output:
  format: [json, markdown, csv]
  dir: ./results/
  
hallucination:
  method: claim_verification
  verifier: gpt-4o
```

## CI Integration

```yaml
# .github/workflows/llm-bench.yml
name: Weekly LLM Benchmark
on:
  schedule:
    - cron: '0 9 * * 1'  # every Monday 9am
jobs:
  bench:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install llm-bench
      - run: llm-bench run --suite general --output results/
      - uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: results/
```

## Roadmap

- [x] Multi-provider benchmarking
- [x] Hallucination detection
- [x] CLI + Python API
- [x] Markdown/JSON/CSV export
- [ ] Web dashboard (live results)
- [ ] Historical tracking (regression alerts)
- [ ] Custom judge models
- [ ] Token-level latency profiling

## License

MIT
