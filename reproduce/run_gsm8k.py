import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from cot_pilot.core.experiment_manager import ExperimentManager

def main():
    parser = argparse.ArgumentParser(description="Run GSM8K Evaluation and Optimization")
    parser.add_argument('--model', '-m', type=str, required=True, help='Model name (e.g., qwen3:0.6b)')
    parser.add_argument('--sample-ratio', type=float, default=0.02, help='Sampling ratio (default 2%)')
    parser.add_argument('--min-samples', type=int, default=10, help='Minimum samples')
    parser.add_argument('--iterations', type=int, default=5, help='Max evolution iterations')
    parser.add_argument('--pop-size', type=int, default=10, help='Population size')
    
    args = parser.parse_args()
    
    manager = ExperimentManager(work_dir="./workspace/gsm8k")
    
    manager.run_experiment(
        dataset_name="gsm8k_gen", 
        model_type=args.model, 
        sample_ratio=args.sample_ratio,
        min_samples=args.min_samples,
        iterations=args.iterations,
        pop_size=args.pop_size
    )

if __name__ == "__main__":
    main()
