import os
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

class Reporter:
    """
    Generates a Markdown report for the experiment.
    """
    def __init__(self, work_dir: str = "./workspace"):
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        self.report_path = os.path.join(self.work_dir, "experiment_report.md")
        
        # Initialize report file (Overwrite existing)
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write("# CoT-Pilot Experiment Report\n\n")
            f.write(f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

    def add_config(self, config: Dict[str, Any]):
        """Adds configuration details to the report."""
        content = "## 1. Configuration\n\n"
        content += "| Parameter | Value |\n|---|---|\n"
        for k, v in config.items():
            content += f"| {k} | {v} |\n"
        content += "\n"
        self._append_to_file(content)

    def add_baseline(self, baseline_data: Dict[str, Any]):
        """Adds baseline comparison results."""
        content = "## 2. Baseline Analysis\n\n"
        content += "### Performance Comparison\n\n"
        content += "| Method | Score |\n|---|---|\n"
        content += f"| Standard Prompting | {baseline_data.get('std_score', 'N/A')} |\n"
        content += f"| Zero-shot CoT | {baseline_data.get('cot_score', 'N/A')} |\n\n"
        
        content += "### Sample Analysis\n\n"
        content += f"- **Improved Samples**: {baseline_data.get('improved_count', 0)}\n"
        content += f"- **Regressed Samples**: {baseline_data.get('regressed_count', 0)}\n"
        content += f"- **Unchanged Samples**: {baseline_data.get('same_count', 0)}\n\n"
        
        if baseline_data.get('improved_samples'):
            content += "**Example Improved IDs**: " + ", ".join(map(str, baseline_data['improved_samples'][:10])) 
            if len(baseline_data['improved_samples']) > 10:
                content += "..."
            content += "\n\n"
        
        # Add detailed sample diff table if available
        sample_diffs = baseline_data.get('sample_diffs', [])
        if sample_diffs:
            content += "### Sample Change Log (Standard vs CoT)\n\n"
            content += "| ID | Status | Question | Standard Answer | CoT Answer |\n"
            content += "|---|---|---|---|---|\n"
            
            for diff in sample_diffs:
                # Sanitize strings to avoid breaking markdown table
                q = str(diff.get('question', '')).replace('\n', ' ')[:50] + "..."
                std = str(diff.get('std_ans', '')).replace('\n', ' ')[:50] + "..."
                cot = str(diff.get('cot_ans', '')).replace('\n', ' ')[:50] + "..."
                status = diff.get('status', '')
                pid = diff.get('id', '')
                
                content += f"| {pid} | {status} | {q} | {std} | {cot} |\n"
            content += "\n"
            
        self._append_to_file(content)

    def add_optimization_step(self, step_data: Dict[str, Any]):
        """Logs a single optimization step (can be called multiple times)."""
        # If this is the first step, write the table header
        if "generation" in step_data and step_data["generation"] == 0:
             header = "## 3. Optimization Trajectory\n\n"
             header += "| Generation | Best Score | Avg Score | Best Prompt |\n|---|---|---|---|\n"
             self._append_to_file(header)
        
        # Truncate prompt for display
        best_prompt = step_data.get('best_prompt', '').replace('\n', ' ')
        if len(best_prompt) > 50:
            best_prompt_disp = best_prompt[:47] + "..."
        else:
            best_prompt_disp = best_prompt
            
        row = f"| {step_data.get('generation')} | {step_data.get('best_score'):.4f} | {step_data.get('avg_score'):.4f} | {best_prompt_disp} |\n"
        self._append_to_file(row)
        
        # Add detailed population table for this generation
        survivors = step_data.get('survivors', step_data.get('population', []))
        candidates = step_data.get('candidates', [])
        
        pop_content = f"\n**Generation {step_data.get('generation')} Details**:\n\n"
        
        # 1. Survivors Table
        pop_content += "#### Survivors (Elite Population)\n"
        pop_content += "| Prompt | Score |\n|---|---|\n"
        for p in survivors:
            p_text = p.get('prompt', '').replace('\n', ' ')
            if len(p_text) > 100:
                p_text = p_text[:97] + "..."
            pop_content += f"| {p_text} | {p.get('score'):.4f} |\n"
        pop_content += "\n"
        
        # 2. Candidates Table (The rejected ones, if any)
        if candidates:
            pop_content += "#### Rejected Candidates (Tried but failed to beat elite)\n"
            pop_content += "| Prompt | Score |\n|---|---|\n"
            for p in candidates:
                p_text = p.get('prompt', '').replace('\n', ' ')
                if len(p_text) > 100:
                    p_text = p_text[:97] + "..."
                pop_content += f"| {p_text} | {p.get('score'):.4f} |\n"
            pop_content += "\n"
            
        self._append_to_file(pop_content)

    def add_final_result(self, result: Dict[str, Any]):
        """Adds final results and conclusion."""
        content = "## 4. Final Results\n\n"
        content += "### Best Found Prompt\n\n"
        content += f"```text\n{result.get('best_prompt')}\n```\n\n"
        
        content += "### Final Metrics\n\n"
        content += f"- **Final Score**: {result.get('final_score'):.4f}\n"
        content += f"- **Improvement over Baseline**: {result.get('improvement'):.4f}\n"
        
        self._append_to_file(content)

        # Add detailed sample diff table for Final Result (Optimized vs Baseline CoT)
        sample_diffs = result.get('sample_diffs', [])
        if sample_diffs:
            content = "### Optimization Impact Analysis (Baseline CoT vs Optimized)\n\n"
            content += "| ID | Status | Question | Baseline CoT Answer | Optimized Answer |\n"
            content += "|---|---|---|---|---|\n"
            
            for diff in sample_diffs:
                # Sanitize strings to avoid breaking markdown table
                q = str(diff.get('question', '')).replace('\n', ' ')[:50] + "..."
                std = str(diff.get('std_ans', '')).replace('\n', ' ')[:50] + "..." # Here std_ans is baseline CoT
                cot = str(diff.get('cot_ans', '')).replace('\n', ' ')[:50] + "..." # Here cot_ans is optimized
                status = diff.get('status', '')
                pid = diff.get('id', '')
                
                content += f"| {pid} | {status} | {q} | {std} | {cot} |\n"
            content += "\n"
            self._append_to_file(content)

    def add_reproducibility(self, command: str):
        """Adds command to reproduce the experiment."""
        content = "## 5. Reproducibility\n\n"
        content += "Run the following command to reproduce this experiment:\n\n"
        content += f"```bash\n{command}\n```\n"
        self._append_to_file(content)

    def _append_to_file(self, content: str):
        """Appends content to the report file."""
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(content)

    # Redefining methods to be safe for streaming
    
    def init_report(self):
        """Writes the initial header."""
        pass # Already done in __init__

    def write_section_header(self, title: str):
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"\n## {title}\n\n")

    def write_content(self, content: str):
         with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(content)
