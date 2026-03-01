import os
import sys
import argparse
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from cot_pilot.core.baseline_analyzer import BaselineAnalyzer
from cot_pilot.core.reporter import Reporter

def setup_logger():
    logger = logging.getLogger("Analyzer")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def find_latest_pred_file(base_dir):
    """Recursively finds the latest JSON prediction file in a directory."""
    if not os.path.exists(base_dir):
        return None
        
    latest_file = None
    latest_time = 0
    
    # Priority: results > predictions
    # First check for 'results' directory if it exists
    results_dir = os.path.join(base_dir, "results")
    if not os.path.exists(results_dir):
        # Maybe base_dir IS the parent of timestamps, so we need to search deeper
        pass
        
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Look for JSON files in 'results' or 'predictions' folders
            if file.endswith(".json") and ("predictions" in root or "results" in root):
                # Prefer 'results' over 'predictions' if timestamps are similar?
                # Actually, 'results' contains the processed correctness, 'predictions' usually just raw.
                # So we should filter for 'results' if available.
                
                path = os.path.join(root, file)
                
                # If we already found a file in 'results', ignore 'predictions' unless it's newer
                # But 'results' is always better.
                
                # Let's enforce: MUST be in 'results' if available?
                # But some datasets might not have it.
                
                mtime = os.path.getmtime(path)
                
                # Simple logic: keep latest file, but prefer 'results' over 'predictions' for same timestamp
                # Actually, OpenCompass generates both. 'results' is better.
                
                is_result = "results" in root
                
                if mtime > latest_time:
                    latest_time = mtime
                    latest_file = path
                elif mtime == latest_time and is_result:
                     latest_file = path # Upgrade to result file if same time
                     
    # Second pass: If we found a file in 'predictions', check if a corresponding 'results' exists
    if latest_file and "predictions" in latest_file:
        possible_result = latest_file.replace("predictions", "results")
        if os.path.exists(possible_result):
            return possible_result
            
    return latest_file

def analyze_experiment(work_dir, output_file=None):
    logger = setup_logger()
    work_dir = os.path.abspath(work_dir)
    
    logger.info(f"🔍 Analyzing experiment in: {work_dir}")
    
    # 1. Locate Baseline CoT Results
    baseline_dir = os.path.join(work_dir, "baseline", "cot")
    baseline_pred = find_latest_pred_file(baseline_dir)
    
    if not baseline_pred:
        logger.error(f"❌ Could not find Baseline CoT predictions in {baseline_dir}")
        return
    logger.info(f"✅ Found Baseline CoT: {baseline_pred}")
    
    # 2. Locate Optimized Results (Final Evaluation)
    # Usually in opt/eval or we might need to search in specific timestamp folders
    # ExperimentManager runs final eval in a temp dir, but it's usually inside baseline/final or opt/eval?
    # Let's check 'baseline/final' first (from latest code), then 'opt/eval'
    
    final_dir = os.path.join(work_dir, "baseline", "final") # Based on previous logs
    if not os.path.exists(final_dir):
        # Fallback to opt/eval if final wasn't used
        final_dir = os.path.join(work_dir, "opt", "eval")
        
    final_pred = find_latest_pred_file(final_dir)
    
    if not final_pred:
        logger.error(f"❌ Could not find Optimized/Final predictions in {final_dir}")
        return
    logger.info(f"✅ Found Optimized Result: {final_pred}")
    
    # 3. Run Comparison
    analyzer = BaselineAnalyzer(work_dir=work_dir)
    
    # Mock result objects expected by _compare_results
    # We only need the 'pred_file' key and 'score' (optional for diffs)
    baseline_res = {"pred_file": baseline_pred, "score": 0.0} 
    final_res = {"pred_file": final_pred, "score": 0.0}
    
    logger.info("Running comparison...")
    report_data = analyzer._compare_results(baseline_res, final_res)
    
    # 4. Generate Report
    if output_file:
        report_path = output_file
    else:
        report_path = os.path.join(work_dir, "analysis_report.md")
        
    logger.info(f"Improved Samples: {report_data['improved_count']}")
    logger.info(f"Regressed Samples: {report_data['regressed_count']}")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Post-Hoc Optimization Analysis\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Files Analyzed\n")
        f.write(f"- **Baseline**: `{baseline_pred}`\n")
        f.write(f"- **Optimized**: `{final_pred}`\n\n")
        
        f.write("## Impact Summary\n")
        f.write(f"- **Improved**: {report_data['improved_count']} (Wrong -> Correct)\n")
        f.write(f"- **Regressed**: {report_data['regressed_count']} (Correct -> Wrong)\n")
        f.write(f"- **Unchanged**: {report_data['same_count']}\n\n")
        
        sample_diffs = report_data.get('sample_diffs', [])
        if sample_diffs:
            f.write("## Detailed Change Log (Baseline CoT vs Optimized)\n\n")
            f.write("| ID | Status | Question | Baseline CoT Answer | Optimized Answer |\n")
            f.write("|---|---|---|---|---|\n")
            
            for diff in sample_diffs:
                # Sanitize
                q = str(diff.get('question', '')).replace('\n', ' ')[:50] + "..."
                std = str(diff.get('std_ans', '')).replace('\n', ' ')[:50] + "..."
                cot = str(diff.get('cot_ans', '')).replace('\n', ' ')[:50] + "..."
                status = diff.get('status', '')
                pid = diff.get('id', '')
                
                f.write(f"| {pid} | {status} | {q} | {std} | {cot} |\n")
        else:
            f.write("\n*No status changes detected between these two runs.*")
            
    logger.info(f"✨ Report generated: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze existing experiment results.")
    parser.add_argument('work_dir', type=str, help='Path to experiment workspace (e.g., workspace/gsm8k)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output report path')
    
    args = parser.parse_args()
    analyze_experiment(args.work_dir, args.output)
