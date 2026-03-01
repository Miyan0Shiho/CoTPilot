import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from cot_pilot.core.experiment_manager import ExperimentManager

def main():
    parser = argparse.ArgumentParser(description="Run AIME2024 Full Evaluation and Optimization")
    parser.add_argument('--model', '-m', type=str, required=True, help='Model name (e.g., qwen3:0.6b)')
    parser.add_argument('--iterations', type=int, default=10, help='Max evolution iterations')
    parser.add_argument('--pop-size', type=int, default=5, help='Population size')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    
    args = parser.parse_args()
    
    manager = ExperimentManager(work_dir="./workspace/aime2024_full")
    
    # Run AIME2024 with sample_ratio=1.0 (Full Dataset)
    manager.run_experiment(
        dataset_name="aime2024_gen", 
        model_type=args.model, 
        sample_ratio=1.0, 
        iterations=args.iterations,
        pop_size=args.pop_size,
        patience=args.patience
    )

if __name__ == "__main__":
    main()
