import sys
import os
import json
from reproduce_results import run_reproduction

def run_benchmarks():
    datasets = [
        "aime2024",
        "mmlu_demo", # Using mmlu_demo or similar small subset for quick test if full mmlu is too big. 
                     # Wait, mmlu_gen loads the full MMLU usually. 
                     # Let's use 'mmlu' (the abbr in config) -> actually the config name is usually passed to DatasetManager.
                     # DatasetManager maps 'mmlu_gen' to the config module.
                     # Let's check what names work.
        "humaneval",
        "cmmlu"
    ]
    
    # Map friendly names to likely config names if needed
    # Based on previous LS, we have:
    # aime2024 -> aime2024_gen
    # mmlu -> mmlu_gen
    # humaneval -> humaneval_gen
    # cmmlu -> cmmlu_gen
    
    benchmarks = {
        "aime2024": "aime2024_gen",
        "mmlu_demo": "mmlu_demo",
        "cmmlu_demo": "cmmlu_demo"
    }
    
    # We will print results after all runs, so we need to collect them.
    # But let's print as we go to be safe.
    
    results = {}
    
    with open("benchmark_report.md", "w") as f:
        f.write("# CoT-Pilot Benchmark Report\n\n")
        f.write("| Dataset | Status | Baseline (Std) | Baseline (CoT) | Improved Samples | Best Prompt |\n")
        f.write("|---------|--------|----------------|----------------|------------------|-------------|\n")
    
    print("\n# CoT-Pilot Benchmark Report\n")
    print("| Dataset | Status | Baseline (Std) | Baseline (CoT) | Improved Samples | Best Prompt |")
    print("|---------|--------|----------------|----------------|------------------|-------------|")
    
    for name, config_name in benchmarks.items():
        # Capture stdout/stderr to keep table clean? Maybe too complex.
        # Just run it.
        try:
            # Run with very small sample for speed
            # Note: run_reproduction prints a lot of logs. 
            # We can't easily suppress them without redirecting stdout.
            # Let's just run it and print the summary line at the end of each run.
            
            res = run_reproduction(
                dataset=config_name, 
                model="qwen3:0.6b", 
                sample_ratio=0.01, # 1% or min 5 samples
                min_samples=5, 
                iterations=2, 
                pop_size=2
            )
            
            if res['status'] == 'success':
                status = "✅ Success"
                if res['is_optimized']:
                    status += " (Optimized)"
                std = f"{res['baseline_std']:.2f}"
                cot = f"{res['baseline_cot']:.2f}"
                imp = res['improved_samples']
                prompt = res['best_prompt'][:50].replace("\n", " ") + "..." if len(res['best_prompt']) > 50 else res['best_prompt'].replace("\n", " ")
            else:
                status = "❌ Failed"
                std = "-"
                cot = "-"
                imp = "-"
                prompt = f"Error: {res.get('error', 'Unknown')}"
            
            # Print the row
            row = f"| {name} | {status} | {std} | {cot} | {imp} | {prompt} |"
            print(row)
            with open("benchmark_report.md", "a") as f:
                f.write(row + "\n")
                
            results[name] = res
            
        except Exception as e:
            err_row = f"| {name} | ❌ Critical Error | - | - | - | {str(e)} |"
            print(err_row)
            with open("benchmark_report.md", "a") as f:
                f.write(err_row + "\n")
            
    print("\n## Detailed Report generated.")

if __name__ == "__main__":
    # Ensure we can import modules
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    run_benchmarks()
