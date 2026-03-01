from abc import ABC, abstractmethod
import random
import math
from typing import List, Any, Dict, Optional
from collections import defaultdict

class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, data: List[Any], **kwargs) -> List[Any]:
        pass

class RandomSampling(SamplingStrategy):
    """
    Standard random sampling with ratio and min/max constraints.
    """
    def sample(self, data: List[Any], ratio: float = 0.05, min_samples: int = 20, max_samples: int = 200, seed: int = 42, **kwargs) -> List[Any]:
        n = len(data)
        if n == 0:
            return []
            
        target_n = int(n * ratio)
        target_n = max(min_samples, target_n)
        target_n = min(max_samples, target_n)
        target_n = min(n, target_n)
        
        random.seed(seed)
        return random.sample(data, target_n)

class StratifiedSampling(SamplingStrategy):
    """
    Stratified sampling based on a grouping key (e.g., 'subject').
    Ensures each group is represented.
    """
    def sample(self, data: List[Any], key: str = 'subject', samples_per_group: int = 2, ratio: float = 0.05, max_total: int = 300, seed: int = 42, **kwargs) -> List[Any]:
        if not data:
            return []
            
        # Group data
        groups = defaultdict(list)
        for item in data:
            # Fallback if key missing
            group_val = item.get(key, 'unknown')
            groups[group_val].append(item)
            
        sampled_data = []
        random.seed(seed)
        
        # 1. Base allocation: Ensure minimum representation per group
        for group_val, items in groups.items():
            # If group is small, take all
            k = min(len(items), samples_per_group)
            picked = random.sample(items, k)
            sampled_data.extend(picked)
            
        # 2. Proportional fill-up
        current_count = len(sampled_data)
        total_target = min(len(data), max(int(len(data) * ratio), max_total))
        
        remaining_slots = total_target - current_count
        if remaining_slots > 0:
            # Flatten remaining candidates
            # Note: This is a simplified approach. Ideally we remove already picked ones.
            # But since we need ID or object identity to deduplicate and dicts are not hashable...
            # We can use a simple trick: use indices
            
            # Re-implement using indices
            all_indices = list(range(len(data)))
            # Mark picked indices
            # This requires mapping back from groups to original list, which is complex.
            # Easier: Just sample from the pool of (all - picked).
            
            # Let's restart with a cleaner index-based approach
            group_indices = defaultdict(list)
            for idx, item in enumerate(data):
                group_val = item.get(key, 'unknown')
                group_indices[group_val].append(idx)
            
            picked_indices = set()
            
            # Phase 1
            for g_idxs in group_indices.values():
                k = min(len(g_idxs), samples_per_group)
                selected = random.sample(g_idxs, k)
                picked_indices.update(selected)
                
            # Phase 2
            remaining_indices = [i for i in range(len(data)) if i not in picked_indices]
            remaining_slots = max(0, total_target - len(picked_indices))
            
            if remaining_indices and remaining_slots > 0:
                k = min(len(remaining_indices), remaining_slots)
                extra_picked = random.sample(remaining_indices, k)
                picked_indices.update(extra_picked)
            
            # Reconstruct list
            return [data[i] for i in sorted(list(picked_indices))]
            
        return sampled_data

class LogScalingSampling(SamplingStrategy):
    """
    Adaptive sampling where sample size scales logarithmically with dataset size.
    Good for very large datasets to avoid linear growth.
    Formula: N = multiplier * log(total_size) ^ power
    Or simple sqrt scaling.
    """
    def sample(self, data: List[Any], multiplier: float = 10.0, min_samples: int = 20, max_samples: int = 500, seed: int = 42, **kwargs) -> List[Any]:
        n = len(data)
        if n == 0:
            return []
            
        # Example: N = 10 * sqrt(n)
        # 100 -> 100
        # 1000 -> 316
        # 10000 -> 1000 (capped)
        
        # Let's use sqrt scaling as it's more aggressive than linear but less than log
        # Or just use the user's request: "Adaptive"
        
        # Target = min_samples + (max_samples - min_samples) * (1 - e^(-n/k)) ?
        # Simple Square Root Scaling
        raw_target = int(multiplier * math.sqrt(n))
        
        target_n = max(min_samples, raw_target)
        target_n = min(max_samples, target_n)
        target_n = min(n, target_n)
        
        random.seed(seed)
        return random.sample(data, target_n)

class StrategyFactory:
    _strategies = {
        "random": RandomSampling(),
        "stratified": StratifiedSampling(),
        "log_scaling": LogScalingSampling()
    }
    
    @classmethod
    def get(cls, name: str) -> SamplingStrategy:
        return cls._strategies.get(name, cls._strategies["random"])
