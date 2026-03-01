import os
import argparse
from typing import List, Dict, Any

from cot_pilot.core.evoprompt_adapter import CoTEvaluator, CustomGAEvoluter, CustomDEEvoluter

class Optimizer:
    """
    Wraps EvoPrompt to optimize prompts.
    """
    
    def __init__(self, work_dir: str = "./workspace/opt"):
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        
    def optimize(self, data: List[Dict[str, Any]], initial_prompt: str, 
                 task_type: str = "qa", algo: str = "ga", 
                 pop_size: int = 5, iteration: int = 3,
                 openai_key: str = None, model_type: str = "gpt-3.5-turbo",
                 dataset_name: str = "gsm8k_gen", concurrency_params: Dict[str, Any] = None,
                 progress_callback=None, target_score=None, patience: int = None) -> str:
        """
        Runs optimization.
        """
        if concurrency_params is None:
            concurrency_params = {}
        
        # 2. Mock Arguments
        args = argparse.Namespace()
        args.task = task_type
        args.dataset = "custom_pilot"
        args.dev_file = None 
        args.test_file = None
        
        if "gpt" not in model_type.lower():
            os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
            os.environ["OPENAI_API_KEY"] = "EMPTY"
            args.llm_type = model_type 
            args.language_model = model_type
        else:
            args.llm_type = model_type
            args.language_model = model_type
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key

        args.popsize = pop_size
        args.budget = iteration * pop_size 
        args.iteration = iteration
        args.prompt_num = 0 
        args.sample_num = len(data)
        args.openai_api_key = openai_key or os.environ.get("OPENAI_API_KEY")
        args.position = "zero"
        args.output = self.work_dir
        
        # EvoPrompt params
        args.alpha = 0.1
        args.mu = 0.0
        args.sigma = 0.0
        args.initial = "manual" # Use our prompt
        args.sel_mode = "wheel"
        args.ga_mode = "topk"
        args.write_step = 1
        args.max_new_tokens = 2048 # For LLM client if needed

        # 3. Initialize Evaluator
        evaluator = CoTEvaluator(args, self.work_dir, dataset_name=dataset_name, concurrency_params=concurrency_params)
        evaluator.load_data(data)
        
        # 4. Initialize Evoluter
        try:
            if algo == 'ga':
                runner = CustomGAEvoluter(args, evaluator, initial_prompt, progress_callback, target_score=target_score, patience=patience)
            else:
                runner = CustomDEEvoluter(args, evaluator, initial_prompt, progress_callback, target_score=target_score, patience=patience)
                
            runner.evolute()
            
            # Extract the best prompt from runner.population or runner.evaluated_prompts
            # After evolution, runner.population is sorted by score (if using GAEvoluter)
            if runner.population:
                return runner.population[0]
            else:
                return initial_prompt
            
        except StopIteration:
            # Target score reached
            if runner.population:
                return runner.population[0]
            else:
                return initial_prompt
        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return initial_prompt
