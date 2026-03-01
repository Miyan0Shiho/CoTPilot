import argparse
import sys
import os

# 1. Setup Environment Paths
# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add third_party dependencies to path
third_party_dir = os.path.join(current_dir, "third_party")
if os.path.exists(third_party_dir):
    sys.path.insert(0, os.path.join(third_party_dir, "opencompass"))
    sys.path.insert(0, os.path.join(third_party_dir, "EvoPrompt"))

from cot_pilot.core.experiment_manager import ExperimentManager

def main():
    parser = argparse.ArgumentParser(description="CoT-Pilot: Automated Chain-of-Thought Optimization Framework")
    
    # Core Arguments
    parser.add_argument('--dataset', '-d', type=str, required=True, 
                        help='Dataset to optimize for (e.g., gsm8k, mmlu, aime2024)')
    parser.add_argument('--model', '-m', type=str, required=True, 
                        help='Model name supported by OpenCompass (e.g., qwen3:0.6b, hf_llama3_8b)')
    
    # Experiment Parameters
    parser.add_argument('--sample-ratio', type=float, default=0.01, 
                        help='Ratio of dataset to use for optimization (default: 0.01)')
    parser.add_argument('--min-samples', type=int, default=10, 
                        help='Minimum number of samples if ratio yields too few')
    parser.add_argument('--iterations', type=int, default=5, 
                        help='Maximum number of evolution generations')
    parser.add_argument('--pop-size', type=int, default=5, 
                        help='Population size for evolutionary algorithm')
    parser.add_argument('--patience', type=int, default=3, 
                        help='Generations to wait without improvement before stopping')
    
    # Concurrency / Performance
    parser.add_argument('--max-workers', type=int, default=4, 
                        help='Maximum concurrent workers for evaluation')
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='Inference batch size per worker')
    parser.add_argument('--qps', type=float, default=None, 
                        help='Queries per second limit (for API models)')
    
    # Output
    parser.add_argument('--work-dir', type=str, default=None, 
                        help='Custom workspace directory (default: ./workspace/{dataset})')

    args = parser.parse_args()
    
    # Dataset Name Mapping
    # Maps short names to OpenCompass config names
    dataset_map = {
        'gsm8k': 'gsm8k_gen',
        'mmlu': 'mmlu_gen',
        'aime': 'aime2024_gen',
        'aime2024': 'aime2024_gen',
        'humaneval': 'humaneval_gen',
        'bbh': 'bbh_gen'
    }
    
    full_dataset_name = dataset_map.get(args.dataset.lower(), args.dataset)
    
    # Workspace Setup
    if args.work_dir:
        work_dir = args.work_dir
    else:
        # Use dataset short name for folder
        short_name = args.dataset.lower().split('_')[0]
        work_dir = f"./workspace/{short_name}"
        
    print(f"🚀 CoT-Pilot Starting...")
    print(f"📂 Workspace: {work_dir}")
    print(f"📊 Dataset: {full_dataset_name} (Sample Ratio: {args.sample_ratio})")
    print(f"🤖 Model: {args.model}")
    
    # Concurrency Config
    concurrency_params = {
        "max_num_workers": args.max_workers,
        "batch_size": args.batch_size
    }
    if args.qps:
        concurrency_params["query_per_second"] = args.qps
        
    # Initialize Manager
    manager = ExperimentManager(work_dir=work_dir)
    
    # Run
    manager.run_experiment(
        dataset_name=full_dataset_name, 
        model_type=args.model, 
        sample_ratio=args.sample_ratio,
        min_samples=args.min_samples,
        iterations=args.iterations,
        pop_size=args.pop_size,
        patience=args.patience,
        concurrency_params=concurrency_params
    )

if __name__ == "__main__":
    main()
