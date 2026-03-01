import argparse
import os
import sys
import logging
from cot_pilot.core.optimizer import Optimizer
from cot_pilot.core.dataset_manager import DatasetManager
from cot_pilot.core.baseline_analyzer import BaselineAnalyzer

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("Reproduction")

def run_reproduction(dataset, model, sample_ratio=0.01, min_samples=10, iterations=10, pop_size=5):
    logger = setup_logger()
    logger.info(f"Starting reproduction for {dataset} with {model}")
    
    # 1. Load & Sample Data
    logger.info("Step 1/4: Loading and Sampling Dataset...")
    dm = DatasetManager()
    try:
        data, _ = dm.load_dataset(dataset)
        sampled_data = dm.sample_data(data, strategy='random', ratio=sample_ratio, min_samples=min_samples)
        logger.info(f"Using {len(sampled_data)} samples (Original: {len(data)})")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {
            "dataset": dataset,
            "status": "failed",
            "error": str(e)
        }

    # 2. Baseline Analysis
    logger.info("Step 2/4: Analyzing Baseline Performance...")
    analyzer = BaselineAnalyzer()
    initial_prompt = "Let's think step by step."
    
    concurrency_params = {"max_num_workers": 4, "batch_size": 4}
    
    try:
        report = analyzer.analyze(
            sampled_data, 
            dataset_name=dataset, 
            model_type=model, 
            cot_prompt=initial_prompt,
            concurrency_params=concurrency_params
        )
    except Exception as e:
        logger.error(f"Baseline analysis failed: {e}")
        return {
            "dataset": dataset,
            "status": "failed",
            "error": f"Baseline analysis: {str(e)}"
        }
    
    logger.info("Baseline Results:")
    logger.info(f"  Standard Score: {report['std_score']:.4f}")
    logger.info(f"  CoT Score:      {report['cot_score']:.4f}")
    logger.info(f"  Improved:       {report['improved_count']} samples")
    
    target_score = report['cot_score']
    
    # 3. Optimization
    logger.info(f"Step 3/4: Starting Evolution (Target > {target_score:.4f})...")
    optimizer = Optimizer()
    
    try:
        best_prompt = optimizer.optimize(
            data=sampled_data,
            initial_prompt=initial_prompt,
            model_type=model,
            dataset_name=dataset,
            pop_size=pop_size,
            iteration=iterations,
            concurrency_params=concurrency_params,
            target_score=target_score
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return {
            "dataset": dataset,
            "status": "failed",
            "error": f"Optimization: {str(e)}"
        }
    
    # 4. Final Verification (Optional)
    logger.info("Step 4/4: Optimization Finished.")
    logger.info(f"Best Prompt Found: {best_prompt}")
    
    result = {
        "dataset": dataset,
        "status": "success",
        "baseline_std": report['std_score'],
        "baseline_cot": report['cot_score'],
        "improved_samples": report['improved_count'],
        "best_prompt": best_prompt,
        "is_optimized": best_prompt != initial_prompt
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="One-click reproduction of CoT optimization.")
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset name (e.g., gsm8k_gen, mmlu_gen)')
    parser.add_argument('--model', '-m', type=str, required=True, help='Model name (e.g., gpt-3.5-turbo, qwen3:0.6b)')
    parser.add_argument('--sample-ratio', type=float, default=0.01, help='Sampling ratio (0.01 = 1%)')
    parser.add_argument('--min-samples', type=int, default=10, help='Minimum samples to use')
    parser.add_argument('--iterations', type=int, default=10, help='Max evolution iterations')
    parser.add_argument('--pop-size', type=int, default=5, help='Population size')
    
    args = parser.parse_args()
    
    # Run the logic
    run_reproduction(args.dataset, args.model, args.sample_ratio, args.min_samples, args.iterations, args.pop_size)

if __name__ == "__main__":
    # Ensure project root is in path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    main()
