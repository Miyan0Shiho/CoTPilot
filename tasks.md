# Tasks: Scientific Framework Refactoring

## 1. Core Architecture & Logging (Foundational)
- [ ] Create `cot_pilot/core/logger.py` to standardize logging with file output (`experiment.log`).
- [ ] Create `cot_pilot/core/reporter.py` to handle Markdown report generation.
- [ ] Refactor `Evaluator` to return detailed metrics (dict) instead of scalar, supporting `pass@k` or `exact_match`.

## 2. Baseline Analysis & Comparison (Requirement 1, 6)
- [ ] Enhance `BaselineAnalyzer` in `cot_pilot/core/baseline_analyzer.py`:
    - Support comparing arbitrary metrics (not just `score`).
    - Generate a list of "Improved Sample IDs" where CoT > Standard.
    - Write baseline results to the markdown report.

## 3. Optimization Loop Refactoring (Requirement 3)
- [ ] Update `Optimizer` in `cot_pilot/core/optimizer.py`:
    - Implement adaptive stopping condition (`while best_score <= baseline_cot_score`).
    - Add `patience` parameter to stop if no improvement after N generations.
    - Integrate with `Reporter` to log evolution progress.

## 4. Dataset Support & Full Evaluation (Requirement 5)
- [ ] Create `reproduce/run_aime2024.py` for full-scale AIME2024 evaluation.
- [ ] Create `reproduce/run_gsm8k.py` and `reproduce/run_mmlu.py`.
- [ ] Verify AIME2024 configuration in OpenCompass and ensure `math_verify` dependency is handled.

## 5. Experiment Manager & Entry Point (Requirement 4, 7)
- [ ] Create `cot_pilot/core/experiment_manager.py` to orchestrate the full pipeline.
- [ ] Update `cot_pilot/cli.py` to expose the new scientific workflow.
- [ ] Ensure all console output is minimal and informative, redirecting details to `experiment.log`.

## 6. Cleanup & Verification
- [ ] Delete `cot_pilot/webui.py` and remove related dependencies.
- [ ] Run `reproduce/run_gsm8k.py` (small subset) to verify the report generation and logging.
- [ ] Verify that the final Markdown report contains all requested sections (Config, Baseline, Trajectory, Results).
