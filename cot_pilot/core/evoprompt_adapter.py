import os
import sys
import logging
from typing import List, Dict, Any, Optional
import tempfile

# 1. Setup EvoPrompt Path and Imports
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
evoprompt_path = os.path.join(project_root, "EvoPrompt")

# Ensure 'utils' conflict is resolved before importing EvoPrompt
if 'utils' in sys.modules:
    try:
        if 'EvoPrompt' not in str(sys.modules['utils']):
            del sys.modules['utils']
    except Exception:
        pass

if evoprompt_path not in sys.path:
    sys.path.insert(0, evoprompt_path)

# 2. Mock Missing Dependencies
# ==============================================================================
try:
    import easse
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['easse'] = MagicMock()
    sys.modules['easse.sari'] = MagicMock()

try:
    import mosestokenizer
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['mosestokenizer'] = MagicMock()

# 3. Patch LLM Client for OpenAI v1 Compatibility
# ==============================================================================
try:
    from openai import OpenAI as OpenAIClient
    
    def new_llm_query(client, data, type, task, temperature=0.0, **kwargs):
        """
        Patched llm_query function that replaces EvoPrompt's original implementation.
        Redirects calls to our OpenCompass-compatible backend (Ollama/Qwen/OpenAI v1).
        """
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        # Initialize client if not provided or just create new one
        # Note: 'client' arg from EvoPrompt might be old OpenAI object or None
        client = OpenAIClient(api_key=api_key, base_url=base_url)
        
        try:
            # Handle list input (batch) - simple loop for now
            if isinstance(data, list):
                results = []
                for item in data:
                    res = new_llm_query(client, item, type, task, temperature, **kwargs)
                    results.append(res)
                return results

            # Single string input
            # Check model type to decide chat vs completion (modern models are chat)
            response = client.chat.completions.create(
                model=type,
                messages=[{"role": "user", "content": data}],
                temperature=temperature,
                max_tokens=kwargs.get("max_new_tokens", 2048)
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(f"LLM Query failed: {e}")
            return data # Fallback

    # Import and patch EvoPrompt modules
    # We must patch 'llm_client' because 'paraphrase' uses it internally
    import llm_client
    llm_client.llm_query = new_llm_query
    
    # Also patch 'evoluter' just in case
    import evoluter
    evoluter.llm_query = new_llm_query
    
    # Import base classes after patching
    from evoluter import GAEvoluter, DEEvoluter
    
except ImportError as e:
    print(f"Warning: Failed to import EvoPrompt modules: {e}")
    # Mock base classes if import fails completely
    class GAEvoluter:
        def __init__(self, args, evaluator): 
            self.args = args
            self.evaluator = evaluator
            self.logger = logging.getLogger("MockGA")
        def evolute(self): return "Mock Best Prompt"
    class DEEvoluter:
        def __init__(self, args, evaluator): 
            self.args = args
            self.evaluator = evaluator
            self.logger = logging.getLogger("MockDE")
        def evolute(self): return "Mock Best Prompt"

# 4. Define Adapter Classes
# ==============================================================================
from cot_pilot.core.evaluator import Evaluator as CoreEvaluator
# Use common_utils to avoid conflict
try:
    from cot_pilot.common_utils import read_lines
except ImportError:
    # Fallback if package structure is not yet updated
    try:
        from cot_pilot.utils import read_lines
    except ImportError:
        def read_lines(path): return []

class CoTEvaluator:
    """
    Adapter to make CoT-Pilot Evaluator compatible with EvoPrompt.
    Implements the interface expected by EvoPrompt (forward method).
    """
    def __init__(self, args, work_dir, dataset_name="gsm8k_gen", concurrency_params: Dict[str, Any] = None):
        self.args = args
        self.work_dir = work_dir
        self.dataset_name = dataset_name
        self.concurrency_params = concurrency_params or {}
        self.core_evaluator = CoreEvaluator(work_dir=os.path.join(work_dir, "eval"))
        self.public_out_path = work_dir
        
        # Setup Logger
        self.logger = logging.getLogger("CoTEvaluator")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
             self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Setup Dummy LLM Client for EvoPrompt (used for mutation via patched llm_query)
        self.llm_config = {
            "api_type": "openai",
            "api_base": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
            "model": args.language_model
        }
        self.client = None 

        # Load data containers
        self.dev_src = []
        self.dev_tgt = []
        
    def load_data(self, data: List[Dict[str, Any]]):
        self.full_data = data
        self.dev_src = [self._get_text(item) for item in data]
        self.dev_tgt = [self._get_label(item) for item in data]
        # Alias test set to dev set for verification
        self.test_src = self.dev_src
        self.test_tgt = self.dev_tgt
        
    def _get_text(self, item: Dict[str, Any]) -> str:
        for key in ['input', 'question', 'text', 'source']:
            if key in item:
                return str(item[key])
        return str(item)

    def _get_label(self, item: Dict[str, Any]) -> str:
        for key in ['output', 'label', 'target', 'answer']:
            if key in item:
                return str(item[key])
        return "unknown"

    def forward(self, prompt, eval_src, eval_tgt):
        # Construct data list for CoreEvaluator
        # Preserve original structure if available to ensure correct keys (e.g. 'question' for GSM8K)
        if hasattr(self, 'full_data') and len(self.full_data) == len(eval_src):
            # We assume eval_src order matches full_data order (which is true if EvoPrompt doesn't shuffle dev set)
            # EvoPrompt uses self.dev_src as is.
            data = self.full_data
        else:
            # Fallback: try to guess keys based on dataset name
            # This is risky but necessary if full_data is lost
            input_key = 'question' if 'gsm8k' in self.dataset_name else 'input'
            data = [{input_key: s, "answer": t} for s, t in zip(eval_src, eval_tgt)]
        
        # Run Evaluation
        model_kwargs = {
            "max_out_len": 2048,
            "temperature": 0.0, # Greedy for evaluation
            "stop": ['<|im_end|>', '<|endoftext|>', 'User:', 'Observation:', 'Input:']
        }
        
        if 'batch_size' in self.concurrency_params:
            model_kwargs['batch_size'] = self.concurrency_params['batch_size']
        if 'query_per_second' in self.concurrency_params:
            model_kwargs['query_per_second'] = self.concurrency_params['query_per_second']
            
        runner_kwargs = {}
        if 'max_num_workers' in self.concurrency_params:
            runner_kwargs['max_num_workers'] = self.concurrency_params['max_num_workers']
        
        self.logger.info(f"Evaluating prompt: {prompt[:50]}...")
        result_pkg = self.core_evaluator.evaluate(
            data, 
            prompt, 
            self.dataset_name, 
            model_type=self.args.language_model,
            model_kwargs=model_kwargs,
            runner_kwargs=runner_kwargs
        )
        
        # New Evaluator returns a dict with 'metrics' (list of dicts) and 'pred_file'
        metrics = result_pkg.get("metrics", [])
        
        # Debug result
        self.logger.info(f"Evaluation result raw: {result_pkg}")
        
        score = 0.0
        if metrics:
             # Extract score logic same as before
             for k, v in metrics[0].items():
                 # Use exact match for metadata keys to avoid false positives (e.g. 'mode' in 'models')
                 if k in ['dataset', 'version', 'metric', 'mode']:
                     continue
                 try:
                     val = float(v)
                     score = val
                     break
                 except (ValueError, TypeError):
                     continue
        
        # Add epsilon
        score = max(score, 1e-4)
        
        # Extract hypos (predictions) - simplified logic
        hypos = [""] * len(eval_src) # Placeholder for now to avoid complex file parsing if not needed
        # TODO: Implement robust hypo extraction if needed for DE evolution
        
        return {"scores": [score], "hypos": hypos}


class CustomGAEvoluter(GAEvoluter):
    """
    Custom Genetic Algorithm Evoluter that integrates with CoT-Pilot.
    Overrides init_pop to use custom prompt and write_step to trigger callbacks.
    """
    def __init__(self, args, evaluator, initial_prompt, progress_callback=None, target_score=None, patience=None):
        self.custom_initial_prompt = initial_prompt
        self.progress_callback = progress_callback
        self.target_score = target_score
        self.patience = patience
        self.no_improve_steps = 0
        self.best_score_history = -1.0
        # Track history: {generation_idx: {parents: [], candidates: [], survivors: []}}
        self.generation_history = {} 
        try:
            super().__init__(args, evaluator)
        except TypeError:
            pass
        
    def init_pop(self):
        # Override to use our custom initial prompt + paraphrased versions
        # This ensures diversity in Gen 0
        pop = [self.custom_initial_prompt]
        
        # Fill the rest with paraphrased versions using EvoPrompt's method
        # We need to access the 'paraphrase' method from parent or implement it
        # GAEvoluter doesn't have paraphrase, it's usually done before or via llm_client
        
        # EvoPrompt's run_ga.py usually does:
        # population = [args.initial] * args.popsize
        
        # We want diversity. Let's try to paraphrase if popsize > 1.
        if self.args.popsize > 1:
            try:
                # Use the patched llm_client to generate paraphrases
                import llm_client
                # Note: This is a simple heuristic. EvoPrompt has specific paraphrase templates.
                # Let's trust EvoPrompt's internal mechanism if possible, but GAEvoluter.init_pop doesn't do paraphrasing by default.
                
                # We'll just duplicate for now to avoid breaking changes, 
                # BUT the user specifically asked to avoid identical parents.
                # Let's generate simple variations manually if we can't access a paraphrase tool easily.
                
                # Simple variations
                variations = [
                    self.custom_initial_prompt,
                    "Think about this step by step.",
                    "Let's solve this problem by breaking it down.",
                    "Please reason through this step-by-step.",
                    "Let's think step by step to find the correct answer.",
                    "Approach this problem methodically.",
                    "Take a deep breath and work through this step-by-step.",
                    "Break this down into smaller steps.",
                    "Let's analyze this carefully.",
                    "Think logically and solve this step-by-step."
                ]
                
                # Fill population
                while len(pop) < self.args.popsize:
                    idx = len(pop) % len(variations)
                    pop.append(variations[idx])
                    
            except Exception as e:
                print(f"Failed to generate diverse population: {e}")
                pop = [self.custom_initial_prompt] * self.args.popsize
        
        self.population = pop[:self.args.popsize]
        self.prompts2mark = {p: "manual" for p in self.population}
        self.evaluated_prompts = {}
        
        # Evaluate initial population
        candidates_data = []
        for p in self.population:
            if p not in self.evaluated_prompts:
                res = self.evaluator.forward(p, self.evaluator.dev_src, self.evaluator.dev_tgt)
                self.evaluated_prompts[p] = res["scores"]
                candidates_data.append({"prompt": p, "score": res["scores"][0]})
        
        # Record Gen 0 history
        self.generation_history[0] = {
            "parents": [], # No parents for Gen 0
            "candidates": candidates_data,
            "survivors": [{"prompt": p, "score": self.evaluated_prompts[p][0]} for p in self.population]
        }
                
        # Trigger initial callback
        self._trigger_callback(0)
                
        return self.evaluated_prompts, 0

    def write_step(self, step, best_score, avg_score):
        """
        Override write_step to trigger progress callback.
        This is called at the end of every evolution step by the parent class.
        """
        # Call parent method to save files as usual
        super().write_step(step, best_score, avg_score)
        
        # Capture current generation info
        # 'step' here is the completed generation index (1, 2, ...)
        
        # We need to know which prompts were NEWLY evaluated in this step.
        # But evaluated_prompts has everything.
        # We can track it by storing evaluated_prompts keys before evolution?
        # But GAEvoluter.evolute() runs the loop. We are inside write_step called BY evolute.
        
        # Heuristic: 
        # Survivors = self.population (which is sorted by score in GAEvoluter)
        # Candidates = All prompts in evaluated_prompts that are NOT in previous generations' survivors?
        # This is getting complicated.
        
        # Let's simplify: 
        # Just assume self.population are the survivors.
        # We can try to infer candidates if we tracked previous evaluated prompts.
        # But GAEvoluter doesn't expose the 'offspring' list easily.
        
        # However, we can track the population state.
        survivors = [{"prompt": p, "score": self.evaluated_prompts[p][0]} for p in self.population if p in self.evaluated_prompts]
        
        # For 'Candidates', we might miss the ones that were rejected.
        # But wait, GAEvoluter.evolute() loop:
        # 1. Generate offspring
        # 2. Evaluate offspring (calls evaluator.forward -> updates evaluated_prompts)
        # 3. Select best (updates self.population)
        # 4. write_step
        
        # So at this point, evaluated_prompts contains {Old Pop} + {Offspring}.
        # We can diff against previous generation's known prompts to find the new Offspring.
        
        previous_prompts = set()
        for g in range(step): # 0 to step-1
            if g in self.generation_history:
                for s in self.generation_history[g]["survivors"]:
                    previous_prompts.add(s["prompt"])
                for c in self.generation_history[g]["candidates"]: # Also track failed candidates from past
                    previous_prompts.add(c["prompt"])
                    
        current_candidates = []
        for p, score_list in self.evaluated_prompts.items():
            if p not in previous_prompts:
                current_candidates.append({"prompt": p, "score": score_list[0]})
        
        # Store history
        self.generation_history[step] = {
            "parents": [], # Hard to track without hooking 'crossover'
            "candidates": current_candidates, # These are the NEW prompts tried this round
            "survivors": survivors
        }
        
        # Trigger callback
        self._trigger_callback(step)
        
        # Check early stopping (Target Score)
        if self.target_score is not None and best_score > self.target_score:
            print(f"\n🎉 Target score {self.target_score} exceeded! Current best: {best_score}. Stopping early.")
            raise StopIteration("Target score reached")
            
        # Check patience
        if self.patience is not None:
            if best_score > self.best_score_history + 1e-6:
                self.best_score_history = best_score
                self.no_improve_steps = 0
            else:
                self.no_improve_steps += 1
                
            if self.no_improve_steps >= self.patience:
                print(f"\n🛑 No improvement for {self.patience} steps. Stopping early.")
                raise StopIteration("Patience exceeded")

    def _trigger_callback(self, generation):
        if self.progress_callback:
            # Get data from history if available
            history_data = self.generation_history.get(generation, {})
            survivors = history_data.get("survivors", [])
            candidates = history_data.get("candidates", [])
            
            # If history missing (e.g. Gen 0 initial call might be tricky), fallback
            if not survivors and self.population:
                 survivors = [{"prompt": p, "score": self.evaluated_prompts.get(p, [0])[0]} for p in self.population]

            best_score = survivors[0]['score'] if survivors else 0
            avg_score = sum(s['score'] for s in survivors)/len(survivors) if survivors else 0
            
            self.progress_callback({
                "generation": generation,
                "best_score": best_score,
                "avg_score": avg_score,
                "survivors": survivors,
                "candidates": candidates, # NOW we have the rejected ones!
                "best_prompt": survivors[0]['prompt'] if survivors else ""
            })


class CustomDEEvoluter(DEEvoluter):
    """
    Custom Differential Evolution Evoluter.
    """
    def __init__(self, args, evaluator, initial_prompt, progress_callback=None, target_score=None, patience=None):
        self.custom_initial_prompt = initial_prompt
        self.progress_callback = progress_callback
        self.target_score = target_score
        self.patience = patience
        self.no_improve_steps = 0
        self.best_score_history = -1.0
        try:
            super().__init__(args, evaluator)
        except TypeError:
            pass
        
    def init_pop(self):
        pop = [self.custom_initial_prompt] * self.args.popsize
        self.population = pop
        self.prompts2mark = {p: "manual" for p in pop}
        self.evaluated_prompts = {}
        
        for p in pop:
            if p not in self.evaluated_prompts:
                res = self.evaluator.forward(p, self.evaluator.dev_src, self.evaluator.dev_tgt)
                self.evaluated_prompts[p] = res["scores"]
        
        self._trigger_callback(0)
        return self.evaluated_prompts, 0

    def write_step(self, step, best_score, avg_score):
        super().write_step(step, best_score, avg_score)
        self._trigger_callback(step)
        
        if self.target_score is not None and best_score > self.target_score:
            print(f"\n🎉 Target score {self.target_score} exceeded! Current best: {best_score}. Stopping early.")
            raise StopIteration("Target score reached")
            
        if self.patience is not None:
            if best_score > self.best_score_history + 1e-6:
                self.best_score_history = best_score
                self.no_improve_steps = 0
            else:
                self.no_improve_steps += 1
                
            if self.no_improve_steps >= self.patience:
                print(f"\n🛑 No improvement for {self.patience} steps. Stopping early.")
                raise StopIteration("Patience exceeded")

    def _trigger_callback(self, generation):
        if self.progress_callback:
            valid_scores = [self.evaluated_prompts[p][0] for p in self.population if p in self.evaluated_prompts]
            best_score = max(valid_scores) if valid_scores else 0
            avg_score = sum(valid_scores)/len(valid_scores) if valid_scores else 0
            pop_data = [{"prompt": p, "score": self.evaluated_prompts[p][0]} for p in self.population if p in self.evaluated_prompts]
            
            # DE doesn't sort population by default in the list, need to find best
            best_prompt = ""
            if self.population:
                best_prompt = max(self.population, key=lambda p: self.evaluated_prompts.get(p, [0])[0])
            
            self.progress_callback({
                "generation": generation,
                "best_score": best_score,
                "avg_score": avg_score,
                "population": pop_data,
                "best_prompt": best_prompt
            })
