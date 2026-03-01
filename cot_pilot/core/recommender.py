from typing import List, Dict, Any, Tuple

class SamplingRecommender:
    """
    Analyzes dataset metadata and system resources to recommend 
    the optimal sampling strategy and configuration.
    """
    
    def recommend(self, data: List[Dict[str, Any]], system_info: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (strategy_name, strategy_kwargs)
        """
        n = len(data)
        
        # Default fallback
        strategy = "random"
        kwargs = {"ratio": 0.05, "min_samples": 20, "max_samples": 200}
        
        # 1. Check for Multi-subset (Stratified)
        # We assume if 'subject' key exists and has high cardinality, it's stratified
        if n > 0:
            sample_size = min(n, 100)
            sample_keys = [d.get('subject') for d in data[:sample_size] if 'subject' in d]
            
            # If we found subject keys in most items
            if len(sample_keys) > sample_size * 0.8:
                unique_subjects = set(sample_keys)
                # If there are multiple subjects (e.g. > 1), recommend stratified
                if len(unique_subjects) > 1:
                    strategy = "stratified"
                    kwargs = {
                        "key": "subject",
                        "samples_per_group": 2, # Ensure at least 2 per subject
                        "ratio": 0.05,
                        "max_total": 300 # Slightly higher max for stratified
                    }
                    return strategy, kwargs

        # 2. Check for Large Datasets (Log Scaling)
        if n > 5000:
            strategy = "log_scaling"
            kwargs = {
                "multiplier": 5.0, # 5 * sqrt(n)
                "min_samples": 50,
                "max_samples": 500 # Allow more samples for big datasets
            }
            return strategy, kwargs
            
        # 3. Small Datasets
        if n < 100:
            # Just take all of them (ratio=1.0)
            strategy = "random"
            kwargs = {"ratio": 1.0, "min_samples": n, "max_samples": n}
            return strategy, kwargs

        return strategy, kwargs
