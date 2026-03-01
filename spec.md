# CoT-Pilot Framework Scientific Refactoring Specification

This specification outlines the refactoring plan to transform the CoT-Pilot framework into a scientifically rigorous experimental platform suitable for academic publication. The focus is on reproducibility, comprehensive analysis, and transparent reporting.

## 1. Core Architecture Refactoring
- **Objective**: Ensure modularity and scientific rigor in the experimental pipeline.
- **Components**:
    - `ExperimentManager`: Orchestrates the full lifecycle (Setup -> Baseline -> Optimize -> Verify -> Report).
    - `BaselineAnalyzer`: rigorously compares Zero-shot CoT vs. Standard Prompting with statistical significance checks (where applicable) and detailed failure analysis.
    - `AdaptiveOptimizer`: Implements the "beat-the-baseline" logic with configurable stopping criteria (score threshold, patience, max iterations).
    - `ResultReporter`: Generates comprehensive Markdown reports and structured logs.

## 2. Frontend Removal (Requirement 2)
- **Action**: Completely remove all WebUI related code and dependencies.
- **Files to Delete**: `cot_pilot/webui.py`, `cot_pilot/pages/`.
- **Cleanup**: Remove `streamlit` from `requirements.txt` (if present) and CLI commands.

## 3. Advanced Metrics Support (Requirement 6)
- **Objective**: Support metrics beyond Accuracy (e.g., Pass@k, Exact Match, BLEU/ROUGE for text tasks).
- **Implementation**:
    - Update `Evaluator` to parse and return raw metric dictionaries from OpenCompass, not just a single scalar.
    - Allow the user to specify the `primary_metric` for optimization (e.g., `accuracy`, `score`, `pass@1`).

## 4. Scientific Logging & Reporting (Requirement 7)
- **Objective**: Move execution details from ephemeral terminal output to persistent, structured logs.
- **Implementation**:
    - **Logging**: Use `logging` module with `FileHandler` and `StreamHandler`.
        - `experiment.log`: Detailed debug info (API calls, raw model outputs).
        - `console`: High-level progress (Step 1/5, Current Best Score).
    - **Reporting**: `ResultReporter` class to generate `experiment_report.md`.
        - **Section 1: Configuration**: Dataset, Model, Hyperparameters.
        - **Section 2: Baseline Analysis**: Table comparing Standard vs. CoT (Score, Improvement Count, Sample IDs).
        - **Section 3: Optimization Trajectory**: Table/Plot of Best Score vs. Generation.
        - **Section 4: Final Results**: Best Prompt, Final Score, Improvement over Baseline.
        - **Section 5: Reproducibility**: Command to rerun this exact experiment.

## 5. Dataset & Full Evaluation (Requirement 5)
- **Objective**: Support AIME2024 (Full), GSM8K, MMLU with reproduction scripts.
- **Implementation**:
    - `reproduce/` directory with dedicated scripts:
        - `run_aime2024.py`: Full evaluation configuration.
        - `run_gsm8k.py`: Standard benchmark.
        - `run_mmlu.py`: Subject-wise or aggregate evaluation.
    - Ensure `aime2024` uses the full test set as requested.

## 6. Execution Flow (Requirement 1, 3, 4)
The `reproduce_results.py` (or new `run_experiment.py`) will follow this strict sequence:
1.  **Setup**: Initialize Logger, load Dataset config.
2.  **Sample (Optional)**: If `sample_ratio < 1.0`, perform stratified sampling. For AIME2024, skip sampling (use full).
3.  **Baseline Analysis**:
    - Run Standard Prompting.
    - Run Zero-shot CoT (`Let's think step by step.`).
    - **Output**: "Improvement List" (Sample IDs where CoT > Standard).
4.  **Optimization Loop**:
    - Initialize Population (Manual + Paraphrased).
    - **Condition**: `while best_score <= baseline_cot_score + epsilon`:
        - Evolve -> Evaluate -> Select.
        - Log every step.
    - **Stop**: If limit reached or target exceeded.
5.  **Final Verification**: Run the best found prompt on the *Test Set* (if split available) or report Validation Score.
6.  **Report Generation**: Write `experiment_report.md`.

## 7. Directory Structure
```
cot_pilot/
├── core/
│   ├── experiment_manager.py  # Main orchestrator
│   ├── baseline_analyzer.py   # Baseline comparison logic
│   ├── optimizer.py           # Evolutionary algorithm wrapper
│   ├── evaluator.py           # OpenCompass interface
│   ├── reporter.py            # Markdown report generator
│   └── logger.py              # Centralized logging config
├── reproduce/
│   ├── run_aime2024.py
│   ├── run_gsm8k.py
│   └── run_mmlu.py
└── cli.py                     # Updated entry point
```

## 8. Verification Plan
- **Unit Test**: Verify `BaselineAnalyzer` correctly identifies improved samples on dummy data.
- **Integration Test**: Run `run_gsm8k.py` with a small subset to verify the full pipeline (Log generation -> Report creation).
- **Full Run**: Execute AIME2024 (Full) or GSM8K (Full) as a final demonstration.
