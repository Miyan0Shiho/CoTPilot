import os
import sys
import json
import subprocess
import importlib
import re
from typing import List, Dict, Any
import pandas as pd

class Evaluator:
    """
    Wraps OpenCompass to evaluate a prompt on a dataset subset.
    Uses official OpenCompass dataset configurations for evaluation logic.
    """
    
    def __init__(self, work_dir: str = "./workspace"):
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        
    def evaluate(self, data: List[Dict[str, Any]], prompt: str, dataset_name: str, model_type: str = "gpt-3.5-turbo", api_key: str = None, model_kwargs: Dict[str, Any] = None, runner_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Runs evaluation using OpenCompass official flow.
        
        Args:
            data: List of data items
            prompt: User prompt (Zero-shot CoT trigger)
            dataset_name: OpenCompass dataset name
            model_type: Model name (e.g. gpt-3.5-turbo, qwen3:0.6b)
            api_key: API Key
            model_kwargs: Additional model arguments (e.g. max_out_len, temperature, stop, batch_size, query_per_second)
            runner_kwargs: Additional runner arguments (e.g. max_num_workers)
        """
        if model_kwargs is None:
            model_kwargs = {}
        if runner_kwargs is None:
            runner_kwargs = {}
            
        # Set default max_out_len if not provided, to support CoT
        if 'max_out_len' not in model_kwargs:
            model_kwargs['max_out_len'] = 2048

        # 1. Get Dataset Config Module
        from cot_pilot.core.dataset_manager import DatasetManager
        dm = DatasetManager()
        try:
            module_name = dm.get_dataset_config_module(dataset_name)
        except ValueError:
             return {"error": f"Dataset {dataset_name} not found"}
        
        # 2. Inspect reader_cfg to map columns
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            return {"error": f"Failed to import {module_name}: {e}"}

        datasets_var_name = None
        for var in dir(mod):
            if var.endswith("_datasets") or var == "datasets":
                datasets_var_name = var
                break
        
        if not datasets_var_name:
             return {"error": "Config format not supported (no datasets list found)"}
             
        dataset_list = getattr(mod, datasets_var_name)
        
        # Handle LazyObject/LazyAttr
        if hasattr(dataset_list, '_object'):
             dataset_list = dataset_list._object
        
        reader_cfg = dataset_list[0].get('reader_cfg', {})
        
        input_columns = reader_cfg.get('input_columns', ['input'])
        if not input_columns:
            input_columns = ['input']
        input_key = input_columns[0] # Take the first column as primary input
        
        # 3. Dump data
        # We assume 'data' items already have the correct keys because they were loaded using the same config!
        data_path = os.path.join(self.work_dir, "eval_data.jsonl")
        
        # Special handling for MMLU: Ensure we dump input, A, B, C, D, target
        # If 'data' was loaded via DatasetManager using MMLU config, it should have these fields.
        # But if we are running optimization, the data might have been transformed?
        # Optimizer.load_data extracts 'input' and 'answer'. It loses other fields?
        # Yes! Optimizer._get_text only returns the input string.
        # We need to preserve the full item structure in Optimizer or pass it through.
        
        # Evaluator receives 'data' which is list of dicts.
        # If it comes from Optimizer.forward, it is constructed as [{"input": s, "answer": t} ...]
        # This is INSUFFICIENT for MMLU which needs A, B, C, D columns.
        
        # We need to fix this in Evaluator or Optimizer.
        # Ideally Evaluator should handle generic data, but MMLU reader expects specific columns.
        # If we are using MMLU config, we must provide A, B, C, D.
        
        # Hack: If dataset is MMLU, and data items don't have options, we might be in trouble.
        # But wait, we are using the SAME data items we loaded.
        # Let's check Optimizer.forward.
        
        with open(data_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        # 4. Generate Config
        config_path = os.path.join(self.work_dir, "eval_config.py")
        self._generate_config(config_path, module_name, datasets_var_name, data_path, prompt, input_key, model_type, api_key, model_kwargs, runner_kwargs)
        
        # 5. Run OpenCompass
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        opencompass_script = os.path.join(project_root, "opencompass/run.py")
        
        cmd = [
            sys.executable, opencompass_script, config_path,
            "-w", self.work_dir,
            "--debug"
        ]
        
        print(f"Running evaluation with command: {' '.join(cmd)}")
        try:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
                for line in p.stdout:
                    print(line, end='')
            
            if p.returncode != 0:
                 return {"error": f"Return code {p.returncode}"}

        except subprocess.CalledProcessError as e:
            return {"error": str(e)}
            
        # 6. Parse Results
        # OpenCompass output structure:
        # work_dir/
        #   {timestamp}/
        #     summary/summary_{timestamp}.csv  (Metrics)
        #     predictions/{dataset}/{model}.json (Predictions)
        #     results/{dataset}/{model}.json (Detailed metrics, sometimes)
        
        # We need both metrics and predictions path to return a rich object
        
        latest_subdir = self._get_latest_subdir()
        if not latest_subdir:
             return {"error": "No execution directory found"}
             
        # Parse Summary (Metrics)
        metrics = self._parse_summary(latest_subdir)
        
        # Find Prediction File
        pred_file = self._find_prediction_file(latest_subdir)
        
        return {
            "metrics": metrics, # List of dicts (rows from csv)
            "pred_file": pred_file,
            "work_dir": latest_subdir
        }

    def _generate_config(self, config_path: str, module_name: str, datasets_var_name: str, data_path: str, prompt: str, input_key: str, model_type: str, api_key: str, model_kwargs: Dict[str, Any], runner_kwargs: Dict[str, Any]):
        
        # Escape the prompt
        safe_prompt = prompt.replace("'", "\\'").replace("\n", "\\n")
        
        # Construct model kwargs string
        model_kwargs_str = ""
        openai_extra_kwargs = model_kwargs.get('openai_extra_kwargs', {})
        
        # Extract known fields
        max_out_len = model_kwargs.get('max_out_len', 2048)
        temperature = model_kwargs.get('temperature', 0.0)
        stop = model_kwargs.get('stop', [])
        batch_size = model_kwargs.get('batch_size', 4)
        query_per_second = model_kwargs.get('query_per_second', 1)
        
        if stop:
             openai_extra_kwargs['stop'] = stop
             
        # Build extra kwargs string
        openai_extra_kwargs_str = f"openai_extra_kwargs={json.dumps(openai_extra_kwargs)},"
        
        # Extract runner fields
        max_num_workers = runner_kwargs.get('max_num_workers', 16)
        
        config_content = f"""
from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.models import OpenAI, OpenAISDK
from opencompass.runners import LocalRunner

# 1. Import official datasets
with read_base():
    from {module_name} import {datasets_var_name} as original_datasets

# 2. Patch datasets
datasets = []

# Ensure original_datasets is iterable (handle LazyObject)
real_datasets = original_datasets
if hasattr(original_datasets, '_object'):
    real_datasets = original_datasets._object

for d in real_datasets:
    # Point to local data
    d['path'] = '{data_path}'
    
    # Intelligent Prompt Injection
    if 'infer_cfg' in d and 'prompt_template' in d['infer_cfg']:
        template_cfg = d['infer_cfg']['prompt_template']
        if 'template' in template_cfg and 'round' in template_cfg['template']:
            rounds = template_cfg['template']['round']
            if rounds and len(rounds) > 0:
                # Find the user prompt (usually the last HUMAN message)
                last_human_idx = -1
                for i, r in enumerate(rounds):
                    if r.get('role') == 'HUMAN':
                        last_human_idx = i
                
                if last_human_idx != -1:
                    original_prompt = rounds[last_human_idx]['prompt']
                    
                    # 1. Remove existing CoT triggers if any (simple heuristic)
                    clean_prompt = original_prompt.replace("Let's think step by step", "").replace("Think step by step", "")
                    
                    # 2. Append user prompt (CoT trigger)
                    # No extra format instruction for GSM8K as requested
                    
                    new_prompt = f"{{clean_prompt}}\\n{safe_prompt}"
                    
                    # Update the prompt
                    rounds[last_human_idx]['prompt'] = new_prompt

    datasets.append(d)

# 3. Model Config
if "gpt" in "{model_type}".lower():
    models = [
        dict(
            type=OpenAI,
            abbr='{model_type}',
            key='{api_key or "ENV"}',
            meta_template=dict(
                round=[
                    dict(role='HUMAN', api_role='user'),
                    dict(role='BOT', api_role='assistant'),
                ],
            ),
            query_per_second={query_per_second},
            max_out_len={max_out_len},
            max_seq_len=4096,
            batch_size={batch_size},
            temperature={temperature},
        )
    ]
else:
    # Ollama / OpenAISDK
    models = [
        dict(
            type=OpenAISDK,
            path='{model_type}',
            openai_api_base='http://localhost:11434/v1',
            key='EMPTY',
            meta_template=dict(
                round=[
                    dict(role='HUMAN', api_role='user'),
                    dict(role='BOT', api_role='assistant'),
                ],
            ),
            query_per_second={query_per_second},
            max_out_len={max_out_len},
            max_seq_len=4096,
            batch_size={batch_size},
            temperature={temperature},
            {openai_extra_kwargs_str}
        )
    ]

# 4. Runner Config
runner = dict(type=LocalRunner, max_num_workers={max_num_workers}, task=dict(type='OpenICLInferTask'))
"""
        with open(config_path, 'w') as f:
            f.write(config_content)

    def _get_latest_subdir(self):
        subdirs = [os.path.join(self.work_dir, d) for d in os.listdir(self.work_dir) if os.path.isdir(os.path.join(self.work_dir, d))]
        subdirs = [d for d in subdirs if os.path.basename(d)[0].isdigit()]
        if subdirs:
            subdirs.sort(key=os.path.getmtime)
            return subdirs[-1]
        return None

    def _parse_summary(self, subdir):
        summary_dir = os.path.join(subdir, "summary")
        if not os.path.exists(summary_dir):
             return []
        files = [f for f in os.listdir(summary_dir) if f.endswith(".csv")]
        if not files:
             return []
        files.sort(key=lambda x: os.path.getmtime(os.path.join(summary_dir, x)))
        try:
            df = pd.read_csv(os.path.join(summary_dir, files[-1]))
            return df.to_dict(orient="records")
        except Exception:
            return []

    def _find_prediction_file(self, subdir):
        # Recursively search for .json file in 'predictions' folder
        pred_dir = os.path.join(subdir, "predictions")
        if not os.path.exists(pred_dir):
            return None
            
        for root, dirs, files in os.walk(pred_dir):
            for file in files:
                if file.endswith(".json"):
                    return os.path.join(root, file)
        return None
