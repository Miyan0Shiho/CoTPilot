import click
import os
import sys
import subprocess
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

@click.group()
def cli():
    """CoT-Pilot: Automated Chain-of-Thought Prompt Optimization Framework"""
    pass

@cli.command()
@click.option('--dataset', '-d', default='gsm8k_gen', help='Dataset name (e.g. gsm8k_gen, mmlu_gen)')
@click.option('--model', '-m', default='gpt-3.5-turbo', help='Model name (e.g. gpt-3.5-turbo, qwen3:0.6b)')
@click.option('--strategy', '-s', default='auto', help='Sampling strategy (random, stratified, log_scaling, auto)')
@click.option('--pop-size', default=5, help='Population size for evolution')
@click.option('--iteration', default=3, help='Number of evolution iterations')
@click.option('--workers', default=4, help='Number of parallel workers')
@click.option('--batch-size', default=4, help='Inference batch size')
@click.option('--qps', default=1, help='Queries per second')
@click.option('--initial-prompt', '-p', default="Let's think step by step.", help='Initial prompt to start optimization')
@click.option('--sample-kwargs', '-k', multiple=True, help='Additional sampling arguments (e.g. ratio=0.1, min_samples=50)')
def optimize(dataset, model, strategy, pop_size, iteration, workers, batch_size, qps, initial_prompt, sample_kwargs):
    """Run prompt optimization from command line."""
    from cot_pilot.core.optimizer import Optimizer
    from cot_pilot.core.dataset_manager import DatasetManager
    
    print(f"🚀 Starting optimization for {dataset} using {model}")
    print(f"Configuration: Strategy={strategy}, Pop={pop_size}, Iter={iteration}")
    
    # Parse sample kwargs
    kwargs = {}
    for item in sample_kwargs:
        if "=" in item:
            key, val = item.split("=", 1)
            # Try type conversion
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            else:
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass # Keep as string
            kwargs[key] = val
            
    print(f"Sampling Params: {kwargs}")
    
    # 1. Load Data
    dm = DatasetManager()
    try:
        data, _ = dm.load_dataset(dataset)
    except Exception as e:
        print(f"❌ Failed to load dataset {dataset}: {e}")
        return

    # 2. Sample Data
    sampled_data = dm.sample_data(data, strategy=strategy, **kwargs)
    print(f"Dataset: {len(data)} -> {len(sampled_data)} samples (Strategy: {strategy})")
    
    concurrency_params = {
        "max_num_workers": workers,
        "batch_size": batch_size,
        "query_per_second": qps
    }
    
    # 3. Baseline Analysis
    from cot_pilot.core.baseline_analyzer import BaselineAnalyzer
    print("\n📊 Running Baseline Analysis...")
    analyzer = BaselineAnalyzer()
    baseline_report = analyzer.analyze(
        sampled_data, 
        dataset_name=dataset, 
        model_type=model, 
        cot_prompt=initial_prompt,
        concurrency_params=concurrency_params
    )
    
    print("\n--- Baseline Report ---")
    print(f"Standard (No CoT) Score: {baseline_report['std_score']:.4f}")
    print(f"Zero-shot CoT Score:     {baseline_report['cot_score']:.4f}")
    print(f"Improved Samples:        {baseline_report['improved_count']}")
    print(f"Regressed Samples:       {baseline_report['regressed_count']}")
    print(f"Same Samples:            {baseline_report['same_count']}")
    print("-----------------------\n")
    
    # Check if we should proceed (optional logic, but user wants continuous optimization)
    # Target score is the baseline CoT score. We want to beat it.
    target_score = baseline_report['cot_score']
    
    # 4. Optimize
    print(f"🚀 Starting Evolution to beat score: {target_score:.4f}")
    optimizer = Optimizer()

    # initial_prompt is now passed from args
    
    best_prompt = optimizer.optimize(
        data=sampled_data,
        initial_prompt=initial_prompt,
        model_type=model,
        dataset_name=dataset,
        pop_size=pop_size,
        iteration=iteration,
        concurrency_params=concurrency_params,
        target_score=target_score
    )
    
    print("\n✅ Optimization Complete!")
    print(f"Best Prompt: {best_prompt}")

if __name__ == '__main__':
    cli()
