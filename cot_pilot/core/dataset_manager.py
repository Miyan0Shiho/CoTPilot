import os
import pkgutil
import importlib
from typing import List, Dict, Any, Tuple
from opencompass.utils.build import build_dataset_from_cfg
import opencompass.configs.datasets as datasets_pkg
from datasets import Dataset, DatasetDict

class DatasetManager:
    def __init__(self):
        self.datasets_map = {}  # name -> module_path
        self._scan_datasets()
        from .recommender import SamplingRecommender
        from .sampling_strategies import StrategyFactory
        self.recommender = SamplingRecommender()
        self.strategy_factory = StrategyFactory()

    def _scan_datasets(self):
        """Scans opencompass.configs.datasets for available dataset configurations."""
        # print(f"DatasetManager: datasets_pkg path: {datasets_pkg.__path__}")
        if not hasattr(datasets_pkg, '__path__'):
            return
            
        base_path = datasets_pkg.__path__[0]
        # Walk through directories
        # print(f"Scanning {base_path}")
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("_"):
                    # Construct module path
                    rel_path = os.path.relpath(os.path.join(root, file), base_path)
                    module_path = "opencompass.configs.datasets." + rel_path.replace(os.path.sep, ".").replace(".py", "")
                    
                    # Use filename without extension as name
                    name = file.replace(".py", "")
                    # print(f"Found dataset config: {name} at {module_path}")
                    self.datasets_map[name] = module_path

    def list_datasets(self) -> List[str]:
        return sorted(list(self.datasets_map.keys()))

    def get_dataset_config_module(self, name: str) -> str:
        if name not in self.datasets_map:
            raise ValueError(f"Dataset {name} not found.")
        return self.datasets_map[name]

    def sample_data(self, data: List[Any], strategy: str = "auto", **kwargs) -> List[Any]:
        """
        Samples a subset of data using specified or recommended strategy.
        
        Args:
            data: Full dataset list.
            strategy: Strategy name ('random', 'stratified', 'log_scaling', 'auto').
            **kwargs: Arguments passed to the strategy.
        """
        if strategy == "auto":
            # Infer system info? For now just None
            strategy, rec_kwargs = self.recommender.recommend(data)
            # Update kwargs with recommended defaults if not provided
            for k, v in rec_kwargs.items():
                if k not in kwargs:
                    kwargs[k] = v
        
        strat_impl = self.strategy_factory.get(strategy)
        return strat_impl.sample(data, **kwargs)

    def load_dataset(self, name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Loads the dataset and returns list of items + reader_cfg.
        If dataset config contains multiple datasets (e.g. MMLU), it loads and merges all of them.
        """
        module_name = self.get_dataset_config_module(name)
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            raise ImportError(f"Failed to import config module {module_name}: {e}")
        
        # Find the dataset config list variable
        dataset_cfg_list = None
        for var_name in dir(mod):
            if var_name.endswith("_datasets") or var_name == "datasets":
                candidate = getattr(mod, var_name)
                # Handle LazyObject/LazyAttr
                if hasattr(candidate, '_object'):
                     candidate = candidate._object
                     
                if isinstance(candidate, list) and len(candidate) > 0:
                    dataset_cfg_list = candidate
                    break
        
        if not dataset_cfg_list:
            raise ValueError(f"Could not find dataset config list in {module_name}")

        # Use the first dataset's reader_cfg as representative
        first_reader_cfg = dataset_cfg_list[0].get('reader_cfg', {})
        
        all_raw_data = []
        
        # Iterate over ALL dataset configs
        for dataset_cfg in dataset_cfg_list:
            # Build dataset
            try:
                dataset = build_dataset_from_cfg(dataset_cfg)
            except Exception as e:
                print(f"Warning: Failed to build sub-dataset from config: {e}. Skipping.")
                continue
            
            # Extract data
            data_source = None
            if hasattr(dataset, 'test'):
                 data_source = dataset.test
            elif hasattr(dataset, 'dataset'):
                 data_source = dataset.dataset
            elif isinstance(dataset, DatasetDict):
                 if 'test' in dataset:
                     data_source = dataset['test']
                 elif 'validation' in dataset:
                     data_source = dataset['validation']
                 elif 'train' in dataset:
                     data_source = dataset['train']
            elif isinstance(dataset, Dataset):
                 data_source = dataset
            else:
                 # Try iterating
                 data_source = dataset
    
            try:
                # Convert to list to ensure it's loaded
                items = list(data_source)
                all_raw_data.extend(items)
            except Exception as e:
                # Fallback for some custom datasets
                if hasattr(dataset, 'keys') and 'test' in dataset.keys():
                    try:
                         items = list(dataset['test'])
                         all_raw_data.extend(items)
                    except Exception as inner_e:
                        print(f"Warning: Failed to iterate sub-dataset: {inner_e}")
                else:
                    print(f"Warning: Failed to iterate sub-dataset: {e}")
            
        if not all_raw_data:
             raise RuntimeError("No data loaded from any sub-dataset.")
             
        return all_raw_data, first_reader_cfg
