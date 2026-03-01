import argparse
import sys
import os
import logging
from cot_pilot.core.dataset_manager import DatasetManager
from cot_pilot.core.baseline_analyzer import BaselineAnalyzer
from cot_pilot.core.optimizer import Optimizer
from cot_pilot.core.reporter import Reporter
from cot_pilot.core.logger import setup_logger

class ExperimentManager:
    """
    Orchestrates the full scientific experiment pipeline.
    """
    
    def __init__(self, work_dir: str = "./workspace/experiment"):
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        self.logger = setup_logger(self.work_dir, "experiment.log")
        self.reporter = Reporter(self.work_dir)
        
    def run_experiment(self, 
                       dataset_name: str, 
                       model_type: str, 
                       sample_ratio: float = 1.0, 
                       min_samples: int = 10, 
                       iterations: int = 10, 
                       pop_size: int = 5,
                       patience: int = 3):
        
        self.logger.info(f"🚀 Starting Experiment: {dataset_name} with {model_type}")
        
        # 0. Log Config
        config = {
            "Dataset": dataset_name,
            "Model": model_type,
            "Sample Ratio": sample_ratio,
            "Min Samples": min_samples,
            "Iterations": iterations,
            "Population Size": pop_size,
            "Patience": patience
        }
        self.reporter.add_config(config)
        
        # 1. Load & Sample Data
        self.logger.info("Step 1/5: Loading and Sampling Dataset...")
        dm = DatasetManager()
        try:
            data, _ = dm.load_dataset(dataset_name)
            
            # Skip sampling if ratio is 1.0 (Full Evaluation)
            if sample_ratio >= 1.0:
                sampled_data = data
                self.logger.info(f"Using FULL dataset: {len(data)} samples")
            else:
                sampled_data = dm.sample_data(data, strategy='random', ratio=sample_ratio, min_samples=min_samples)
                self.logger.info(f"Sampled {len(sampled_data)} samples from {len(data)} (Ratio: {sample_ratio})")
                
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

        # 2. Baseline Analysis
        self.logger.info("Step 2/5: Analyzing Baseline Performance...")
        analyzer = BaselineAnalyzer(work_dir=os.path.join(self.work_dir, "baseline"))
        initial_prompt = "Let's think step by step."
        
        concurrency_params = {"max_num_workers": 4, "batch_size": 4}
        
        # This analyzes Standard vs Baseline CoT
        baseline_report = analyzer.analyze(
            sampled_data, 
            dataset_name=dataset_name, 
            model_type=model_type, 
            cot_prompt=initial_prompt,
            concurrency_params=concurrency_params
        )
        
        self.reporter.add_baseline(baseline_report)
        self.logger.info(f"Baseline CoT Score: {baseline_report.get('cot_score', 0):.4f}")
        
        # Capture the baseline CoT result explicitly for later comparison
        # We re-run _run_eval just to get the result object (it should be fast if cached, but OpenCompass might re-run)
        # To be safe and clean, we will just use the BaselineAnalyzer's internal method if accessible, 
        # or we accept a small overhead.
        # Actually, let's just modify BaselineAnalyzer.analyze to return the raw result objects if possible.
        # But to avoid touching too many files, let's just run it:
        baseline_cot_result = analyzer._run_eval(
            sampled_data, 
            initial_prompt, 
            dataset_name, 
            model_type, 
            concurrency_params, 
            "cot"
        )
        
        target_score = baseline_report.get('cot_score', 0)
        
        # 3. Optimization
        # Force improvement logic: We want to beat the baseline.
        # If patience is hit but we haven't beaten baseline, we might want to continue?
        # Actually, 'patience' in optimizer usually refers to 'no improvement over best found so far'.
        # If best found so far is <= baseline, and we stop, we failed.
        
        # Let's adjust target_score. 
        # If we pass target_score to optimizer, it stops *when it exceeds it*.
        # But we want to find the *best possible*, not just *any* better one.
        # So we should set target_score=1.0 (perfection) or None to let it run until patience/iterations run out.
        # BUT the user wants "until we find a prompt better than baseline".
        # If we set target_score=baseline, it stops immediately upon finding one.
        # The user said: "until optimized out a prompt exceeding baseline... not just stop after few rounds".
        # This implies: KEEP GOING until we find one, OR if we find one, maybe keep going to see if we can do better?
        # Let's interpret "until ... exceeding baseline" as a minimum requirement.
        
        # Strategy:
        # 1. Set target_score = None (don't stop early just because we beat baseline).
        # 2. Rely on 'patience' to stop if we plateau.
        # 3. BUT, if we haven't beaten baseline yet, maybe extend patience?
        # For now, let's stick to standard patience but ensure we log clearly.
        
        self.logger.info(f"Step 3/5: Starting Evolution (Baseline: {target_score:.4f})...")
        optimizer = Optimizer(work_dir=os.path.join(self.work_dir, "opt"))
        
        # Callback to log progress to reporter
        def progress_callback(step_data):
            self.logger.info(f"Gen {step_data['generation']}: Best={step_data['best_score']:.4f}, Avg={step_data['avg_score']:.4f}")
            self.reporter.add_optimization_step(step_data)
            
        best_prompt = optimizer.optimize(
            data=sampled_data,
            initial_prompt=initial_prompt,
            model_type=model_type,
            dataset_name=dataset_name,
            pop_size=pop_size,
            iteration=iterations,
            concurrency_params=concurrency_params,
            target_score=None, # Don't stop immediately when beating baseline, keep optimizing!
            patience=patience,
            progress_callback=progress_callback
        )
        
        # 4. Final Verification (Optional if we trust the optimization result)
        # But rigorous science demands re-evaluating the final prompt to confirm score
        self.logger.info("Step 4/5: Verifying Best Prompt...")
        # Re-run eval with best prompt on the SAME sampled data to be consistent
        # In a real scenario, we might want to run on a held-out test set
        
        # For now, we trust the optimization loop's last evaluation of the best prompt
        # But let's calculate the final improvement
        
        # 5. Final Report
        self.logger.info("Step 5/5: Generating Report...")
        
        # Run final eval with best prompt
        final_eval = analyzer._run_eval(
            sampled_data,
            best_prompt,
            dataset_name,
            model_type,
            concurrency_params,
            "final"
        )
        final_score = final_eval['score']
        
        # Calculate improvement vs Baseline CoT
        improvement_report = analyzer._compare_results(baseline_cot_result, final_eval)
        
        result_summary = {
            "best_prompt": best_prompt,
            "final_score": final_score,
            "improvement": final_score - target_score,
            "sample_diffs": improvement_report.get('sample_diffs', [])
        }
        self.reporter.add_final_result(result_summary)
        
        # Add reproduction command
        cmd = f"python reproduce/run_{dataset_name.split('_')[0]}.py" # Heuristic
        self.reporter.add_reproducibility(cmd)
        
        self.logger.info(f"Experiment Complete. Report saved to {self.reporter.report_path}")

if __name__ == "__main__":
    # Test run
    manager = ExperimentManager()
    manager.run_experiment("gsm8k_gen", "qwen3:0.6b", sample_ratio=0.01, iterations=2)
