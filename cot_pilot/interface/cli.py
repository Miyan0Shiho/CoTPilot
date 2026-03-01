import sys
import os
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from cot_pilot.core.dataset_manager import DatasetManager
from cot_pilot.core.sampler import SamplerFactory
from cot_pilot.core.evaluator import Evaluator
from cot_pilot.core.optimizer import Optimizer

console = Console()

class CLI:
    def __init__(self):
        self.dm = DatasetManager()
        self.sampler_factory = SamplerFactory()
        self.evaluator = Evaluator()
        self.optimizer = Optimizer()
        self.selected_dataset_name = None
        self.dataset = None
        self.subset = None
        self.dev_subset = None
        self.prompt = None
        
    def start(self):
        console.print(Panel.fit("[bold blue]CoT-Pilot Framework[/bold blue]\nSmall-Data CoT Experiment & Optimization Tool", border_style="blue"))
        
        self._select_dataset()
        self._configure_sampling()
        self._run_baseline()
        
        if Confirm.ask("Do you want to optimize this prompt using EvoPrompt?"):
            self._run_optimization()
        else:
            console.print("[green]Experiment finished.[/green]")

    def _select_dataset(self):
        console.print("[bold]Step 1: Select Dataset[/bold]")
        datasets = self.dm.list_datasets()
        if not datasets:
            console.print("[red]No datasets found via OpenCompass registry. Check installation.[/red]")
            sys.exit(1)
            
        # Simple search
        search = Prompt.ask("Enter dataset name (or part of it) to search", default="gsm8k")
        matches = [d for d in datasets if search.lower() in d.lower()]
        
        if not matches:
            console.print("[red]No matches found.[/red]")
            return self._select_dataset()
            
        table = Table(title="Available Datasets")
        table.add_column("Index", justify="right", style="cyan", no_wrap=True)
        table.add_column("Name", style="magenta")
        
        for i, name in enumerate(matches[:20]):
            table.add_row(str(i), name)
            
        console.print(table)
        if len(matches) > 20:
            console.print(f"...and {len(matches)-20} more.")
            
        idx = IntPrompt.ask("Select dataset index", choices=[str(i) for i in range(len(matches[:20]))])
        self.selected_dataset_name = matches[idx]
        
        console.print(f"[yellow]Loading dataset '{self.selected_dataset_name}'...[/yellow]")
        try:
            # We assume 'test' split usually exists, or we just load whatever is default
            # OpenCompass build_dataset usually returns a dict of splits or a single split
            ds_obj = self.dm.load_dataset(self.selected_dataset_name)
            
            # Try to get a listable object
            if hasattr(ds_obj, 'test'):
                self.dataset = list(ds_obj.test)
            elif isinstance(ds_obj, dict) and 'test' in ds_obj:
                self.dataset = list(ds_obj['test'])
            elif isinstance(ds_obj, (list, tuple)):
                self.dataset = list(ds_obj)
            else:
                # Try iterating
                self.dataset = list(ds_obj)
                
            console.print(f"[green]Loaded {len(self.dataset)} items.[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load dataset: {e}[/red]")
            sys.exit(1)

    def _configure_sampling(self):
        console.print("\n[bold]Step 2: Configure Sampling[/bold]")
        strategies = self.sampler_factory.list_samplers()
        
        table = Table(title="Sampling Strategies")
        table.add_column("Name", style="cyan")
        for s in strategies:
            table.add_row(s)
        console.print(table)
        
        strategy_name = Prompt.ask("Choose strategy", choices=strategies, default="random")
        self.test_size = IntPrompt.ask("Test set size (for verification)", default=20)
        self.dev_size = IntPrompt.ask("Dev set size (for optimization)", default=10)
        
        sampler = self.sampler_factory.get_sampler(strategy_name)
        
        # Sample Test Set
        console.print(f"Sampling {self.test_size} items for TEST...")
        self.subset = sampler.sample(self.dataset, self.test_size)
        
        # Sample Dev Set (exclude test items if possible, but for simplicity we re-sample or sample from remainder)
        # Simple approach: Sample test, then sample dev from remainder
        remaining = [x for x in self.dataset if x not in self.subset] # Note: object identity check might fail if dataset recreates objects
        # Fallback: just sample independently
        if not remaining:
             remaining = self.dataset
             
        console.print(f"Sampling {self.dev_size} items for DEV...")
        self.dev_subset = sampler.sample(remaining, self.dev_size)
        
        console.print(f"[green]Ready: Test={len(self.subset)}, Dev={len(self.dev_subset)}[/green]")

    def _run_baseline(self):
        console.print("\n[bold]Step 3: Baseline Experiment[/bold]")
        self.prompt = Prompt.ask("Enter your CoT Prompt (use \\n for newlines)", default="Let's think step by step.")
        self.prompt = self.prompt.replace("\\n", "\n")
        
        self.model = Prompt.ask("Model Type (OpenAI/HF abbr or Ollama model name)", default="gpt-3.5-turbo")
        
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if "gpt" in self.model.lower() and not self.api_key:
             self.api_key = Prompt.ask("Enter OpenAI API Key", password=True)
        
        console.print("[yellow]Running Baseline Evaluation...[/yellow]")
        result = self.evaluator.evaluate(self.subset, self.prompt, self.model, self.api_key)
        
        console.print(Panel(str(result), title="Baseline Results"))

    def _run_optimization(self):
        console.print("\n[bold]Step 4: Optimization[/bold]")
        algo = Prompt.ask("Evolution Algorithm", choices=["ga", "de"], default="ga")
        pop_size = IntPrompt.ask("Population Size", default=5)
        iteration = IntPrompt.ask("Iterations", default=3)
        
        console.print("[yellow]Running EvoPrompt (this may take a while)...[/yellow]")
        best_prompt = self.optimizer.optimize(
            self.dev_subset, self.prompt, 
            task_type="cls", # TODO: Ask user
            algo=algo,
            pop_size=pop_size,
            iteration=iteration,
            openai_key=self.api_key,
            model_type=self.model
        )
        
        console.print(Panel(best_prompt, title="Optimized Prompt"))
        
        if Confirm.ask("Run verification on Test Set with new prompt?"):
            console.print("[yellow]Running Verification...[/yellow]")
            result = self.evaluator.evaluate(self.subset, best_prompt, self.model, self.api_key)
            console.print(Panel(str(result), title="Verification Results"))
