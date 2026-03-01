import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from cot_pilot.core.experiment_manager import ExperimentManager

def main():
    parser = argparse.ArgumentParser(description="Run MMLU Evaluation and Optimization")
    parser.add_argument('--model', '-m', type=str, required=True, help='Model name (e.g., qwen3:0.6b)')
    parser.add_argument('--dataset', '-d', type=str, default='mmlu_gen', help='Dataset config (default: mmlu_gen)')
    parser.add_argument('--sample-ratio', type=float, default=0.01, help='Sampling ratio (default 1%)')
    parser.add_argument('--min-samples', type=int, default=10, help='Minimum samples')
    parser.add_argument('--iterations', type=int, default=10, help='Max evolution iterations')
    parser.add_argument('--pop-size', type=int, default=5, help='Population size')
    
    args = parser.parse_args()
    
    manager = ExperimentManager(work_dir="./workspace/mmlu")
    
    manager.run_experiment(
        dataset_name=args.dataset, 
        model_type=args.model, 
        sample_ratio=args.sample_ratio,
        min_samples=args.min_samples,
        iterations=args.iterations,
        pop_size=args.pop_size
    )

if __name__ == "__main__":
    main()
