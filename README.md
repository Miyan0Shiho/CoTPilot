# CoT-Pilot: Automated Chain-of-Thought Optimization

CoT-Pilot is an advanced framework for automating the discovery of optimal Chain-of-Thought (CoT) prompts. It leverages evolutionary algorithms to evolve prompts that outperform standard baselines (like "Let's think step by step") on complex reasoning tasks.

## Key Features

- **Scientific Workflow**: Standardized `Baseline -> Evolution -> Verification` loop ensures fair and rigorous comparison.
- **Explainable Evolution**: Detailed reports visualize the entire evolutionary trajectory, including survivors and rejected candidates.
- **Adaptive Optimization**: Uses a "Force Improvement" strategy to push beyond local optima.
- **Detailed Impact Analysis**: Automatically identifies and lists samples that were improved (or regressed) by the optimized prompt.

## Quick Start

### Prerequisites

- **Python**: 3.10+ (Recommended)
- **PyTorch**: 2.0+ (Required for OpenCompass)
- **Transformers**: 4.35+
- **OpenCompass**: Handled automatically by `setup_env.py` (with critical patches applied)

### Installation

```bash
git clone https://github.com/Miyan0Shiho/CoT-Pilot.git
cd CoT-Pilot

# One-step environment setup (Clones dependencies, applies patches, installs packages)
python setup_env.py
```

### Running an Experiment

To reproduce the GSM8K optimization experiment:

```bash
# Run with default settings (2% sample ratio, 5 iterations)
python reproduce/run_gsm8k.py --model qwen3:0.6b

# Run a larger experiment
python reproduce/run_gsm8k.py --model qwen3:0.6b --sample-ratio 0.1 --pop-size 10 --iterations 5
```

### Analyzing Results

The framework automatically generates a detailed Markdown report at `workspace/gsm8k/experiment_report.md`. 

You can also re-analyze existing results without re-running the experiment:

```bash
python analyze_experiment.py workspace/gsm8k
```

## Project Structure

- `cot_pilot/core/`: Core logic (Evaluator, Optimizer, Reporter).
- `reproduce/`: Ready-to-run scripts for standard benchmarks (GSM8K, etc.).
- `workspace/`: Default output directory for logs and reports.

## License

MIT
