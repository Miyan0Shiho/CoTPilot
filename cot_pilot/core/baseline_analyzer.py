import os
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
from cot_pilot.core.evaluator import Evaluator

class BaselineAnalyzer:
    """
    Analyzes the baseline performance of standard prompting vs Zero-shot CoT.
    """
    
    def __init__(self, work_dir: str = "./workspace/baseline"):
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        self.evaluator = Evaluator(work_dir=self.work_dir)
        
        # Setup Logger
        self.logger = logging.getLogger("BaselineAnalyzer")
        if not self.logger.handlers:
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def analyze(self, data: List[Dict[str, Any]], dataset_name: str, model_type: str, 
                cot_prompt: str = "Let's think step by step.", 
                concurrency_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Runs two evaluations:
        1. Standard (No CoT trigger)
        2. Zero-shot CoT
        
        Returns a report comparing the two.
        """
        self.logger.info("Starting Baseline Analysis...")
        self.logger.info(f"Dataset: {dataset_name}, Samples: {len(data)}")
        
        # 1. Run Standard Evaluation (Empty prompt or just the question)
        # Note: OpenCompass usually appends the prompt to the input. 
        # If prompt is empty string, it's standard prompting (Direct Answer).
        self.logger.info("Running Standard Prompting (Direct Answer)...")
        std_results = self._run_eval(data, "", dataset_name, model_type, concurrency_params, "standard")
        
        # 2. Run CoT Evaluation
        self.logger.info(f"Running Zero-shot CoT ('{cot_prompt}')...")
        cot_results = self._run_eval(data, cot_prompt, dataset_name, model_type, concurrency_params, "cot")
        
        # 3. Compare Results
        report = self._compare_results(std_results, cot_results)
        
        return report

    def _run_eval(self, data, prompt, dataset_name, model_type, concurrency_params, tag):
        """Helper to run evaluation and extract detailed results"""
        
        # Pass model params
        model_kwargs = {
            "max_out_len": 2048,
            "temperature": 0.0
        }
        if concurrency_params:
            if 'batch_size' in concurrency_params: model_kwargs['batch_size'] = concurrency_params['batch_size']
            if 'query_per_second' in concurrency_params: model_kwargs['query_per_second'] = concurrency_params['query_per_second']
            
        runner_kwargs = {}
        if concurrency_params and 'max_num_workers' in concurrency_params:
            runner_kwargs['max_num_workers'] = concurrency_params['max_num_workers']

        # We'll use a unique sub-directory for each run to avoid overwriting
        run_dir = os.path.join(self.work_dir, tag)
        evaluator = Evaluator(work_dir=run_dir)
        
        result_pkg = evaluator.evaluate(
            data, prompt, dataset_name, 
            model_type=model_type, 
            model_kwargs=model_kwargs,
            runner_kwargs=runner_kwargs
        )
        
        # New Evaluator returns a dict with 'metrics' (list of dicts) and 'pred_file'
        metrics = result_pkg.get("metrics", [])
        pred_file = result_pkg.get("pred_file")
        
        score = 0.0
        if metrics:
             # Extract score logic same as before
             for k, v in metrics[0].items():
                 if isinstance(v, (int, float)) and 'version' not in k:
                     score = float(v)
                     break
                     
        return {
            "score": score,
            "pred_file": pred_file,
            "summary": metrics
        }

    def _find_latest_prediction_file(self, run_dir):
        # Deprecated: Evaluator now returns this
        return None

    def _compare_results(self, std_results, cot_results):
        """
        Compares two prediction files to find improvement/regression.
        Returns a detailed report with sample-level diffs.
        """
        import json
        
        report = {
            "std_score": std_results["score"],
            "cot_score": cot_results["score"],
            "improved_count": 0,
            "regressed_count": 0,
            "same_count": 0,
            "improved_samples": [],
            "regressed_samples": [],
            "sample_diffs": [] # List of dicts: {id, status, question, std_ans, cot_ans}
        }
        
        # Helper to find result file if pred_file points to predictions (not results)
        def resolve_results_file(path):
            if not path: return None
            # If path contains 'predictions', try to find corresponding 'results'
            if "predictions" in path:
                # Standard OpenCompass structure: .../predictions/dataset/model.json
                # Corresponding result: .../results/dataset/model.json
                # But sometimes structure varies.
                res_path = path.replace("predictions", "results")
                if os.path.exists(res_path):
                    return res_path
            return path
            
        std_path = resolve_results_file(std_results["pred_file"])
        cot_path = resolve_results_file(cot_results["pred_file"])
        
        if not std_path or not cot_path:
            self.logger.warning(f"Could not find prediction files to compare details. Std: {std_path}, CoT: {cot_path}")
            return report
            
        try:
            # OpenCompass prediction format is usually a dict mapping index to result or a list
            with open(std_path, 'r') as f:
                std_preds_raw = json.load(f)
            with open(cot_path, 'r') as f:
                cot_preds_raw = json.load(f)
            
            # Normalize to dict mapping ID -> Item
            def normalize_preds(raw_data):
                normalized = {}
                # Case 1: Dict with 'details' list (OpenCompass standard result)
                if isinstance(raw_data, dict) and "details" in raw_data and isinstance(raw_data["details"], list):
                    for idx, item in enumerate(raw_data["details"]):
                        # ID might be in 'example_abbr' or just index
                        pid = str(idx) 
                        normalized[pid] = item
                        # Add a flag for correct extraction later
                        # 'is_correct' is tricky here. OpenCompass details usually have 'pred' and 'answer'.
                        # We need to infer correctness if not present.
                        # But wait, 'accuracy' is top-level.
                        # The details usually don't have 'is_correct'.
                        # Let's assume exact match if is_correct is missing?
                        # No, GSM8K is numeric match.
                        
                        # Actually, looking at the file content:
                        # "pred": ["1450"], "answer": ["2280"]
                        # It doesn't say if it's correct.
                        # However, since we are post-analyzing, we might need to trust the 'accuracy' score but we can't get per-sample correctness easily without re-eval logic.
                        
                        # WAIT. The user wants to see which samples CHANGED.
                        # If we don't have per-sample correctness in the file, we can't do this.
                        # Does OpenCompass result file have per-sample correctness?
                        # I only saw "pred" and "answer".
                        
                        # Let's check if there are other keys in 'details' items.
                        pass
                
                # Case 2: List of items
                elif isinstance(raw_data, list):
                    for idx, item in enumerate(raw_data):
                        normalized[str(idx)] = item
                
                # Case 3: Dict mapping ID to Item
                elif isinstance(raw_data, dict):
                    normalized = {str(k): v for k, v in raw_data.items()}
                    
                return normalized

            std_preds = normalize_preds(std_preds_raw)
            cot_preds = normalize_preds(cot_preds_raw)
            
            # Helper to access items whether list or dict
            def get_item(preds, pid):
                if isinstance(preds, list):
                    # Assuming list is ordered and index matches (risky if shuffled)
                    # Better to rely on 'idx' field if present, or assume order.
                    # For now, if list, we try integer index
                    try:
                        return preds[int(pid)]
                    except:
                        return None
                return preds.get(pid)

            # Determine keys
            if isinstance(std_preds, list):
                common_ids = [str(i) for i in range(len(std_preds))]
            else:
                common_ids = set(std_preds.keys()) & set(cot_preds.keys())
            
            for pid in common_ids:
                std_item = get_item(std_preds, pid)
                cot_item = get_item(cot_preds, pid)
                
                if not std_item or not cot_item:
                    # In some cases item might be 0.0 (False), so check for None explicitly
                    if std_item is None or cot_item is None:
                        continue
                
                # Try to extract correctness
                # Case A: Item is a dict (standard)
                if isinstance(std_item, dict):
                    std_correct = std_item.get('is_correct', std_item.get('score', None))
                    
                    # OpenCompass results format often has "correct": [bool]
                    if std_correct is None and "correct" in std_item and isinstance(std_item["correct"], list):
                         std_correct = std_item["correct"][0]
                    
                    # Special case for GSM8K details: pred list vs answer list fallback
                    if std_correct is None and 'pred' in std_item and 'answer' in std_item:
                         # Simple exact match as fallback? No, GSM8K is robust match.
                         # But we can't implement full eval logic here.
                         # If OpenCompass didn't save 'is_correct', we might be stuck.
                         # However, usually OpenCompass results DO have correctness if accuracy is calculated.
                         # Let's check if 'correct' key exists.
                         std_correct = std_item.get('correct', False) # Sometimes it's boolean 'correct'
                         if isinstance(std_correct, list): std_correct = std_correct[0]
                         
                         # If still missing, we try string exact match (better than nothing for analysis)
                         if 'correct' not in std_item:
                             try:
                                 p = str(std_item['pred'][0]).strip()
                                 a = str(std_item['answer'][0]).strip()
                                 # Remove common artifacts like '####'
                                 p = p.split("####")[-1].strip()
                                 a = a.split("####")[-1].strip()
                                 std_correct = (p == a)
                             except:
                                 std_correct = False
                    
                    # Try to extract content for report
                    question = std_item.get('question', std_item.get('input', std_item.get('origin_prompt', f"Sample {pid}")))
                    std_ans = std_item.get('pred', std_item.get('prediction', 'N/A'))
                    if isinstance(std_ans, list): std_ans = std_ans[0]
                # Case B: Item is a scalar (score directly)
                else:
                    std_correct = std_item
                    # We can't get question/answer from scalar result
                    question = f"Sample ID {pid}"
                    std_ans = "N/A (Scalar Result)"

                if isinstance(cot_item, dict):
                    cot_correct = cot_item.get('is_correct', cot_item.get('score', None))
                    
                    if cot_correct is None and "correct" in cot_item and isinstance(cot_item["correct"], list):
                         cot_correct = cot_item["correct"][0]
                    
                    if cot_correct is None and 'pred' in cot_item and 'answer' in cot_item:
                         cot_correct = cot_item.get('correct', False)
                         if isinstance(cot_correct, list): cot_correct = cot_correct[0]
                         
                         if 'correct' not in cot_item:
                             try:
                                 p = str(cot_item['pred'][0]).strip()
                                 a = str(cot_item['answer'][0]).strip()
                                 p = p.split("####")[-1].strip()
                                 a = a.split("####")[-1].strip()
                                 cot_correct = (p == a)
                             except:
                                 cot_correct = False
                                 
                    if isinstance(std_item, dict):
                        # Use std question if available
                        pass 
                    else:
                        question = cot_item.get('question', cot_item.get('input', cot_item.get('origin_prompt', f"Sample {pid}")))
                    
                    cot_ans = cot_item.get('pred', cot_item.get('prediction', 'N/A'))
                    if isinstance(cot_ans, list): cot_ans = cot_ans[0]
                else:
                    cot_correct = cot_item
                    cot_ans = "N/A (Scalar Result)"
                
                # Convert to boolean
                try:
                    std_correct = bool(float(std_correct))
                except:
                    std_correct = False
                    
                try:
                    cot_correct = bool(float(cot_correct))
                except:
                    cot_correct = False
                
                diff_entry = {
                    "id": pid,
                    "question": question,
                    "std_ans": std_ans,
                    "cot_ans": cot_ans
                }
                
                if not std_correct and cot_correct:
                    report["improved_count"] += 1
                    report["improved_samples"].append(pid)
                    diff_entry["status"] = "✅ Improved"
                    report["sample_diffs"].append(diff_entry)
                elif std_correct and not cot_correct:
                    report["regressed_count"] += 1
                    report["regressed_samples"].append(pid)
                    diff_entry["status"] = "❌ Regressed"
                    report["sample_diffs"].append(diff_entry)
                else:
                    report["same_count"] += 1
                    
        except Exception as e:
            self.logger.error(f"Error comparing files: {e}")
            import traceback
            traceback.print_exc()
            
        return report

