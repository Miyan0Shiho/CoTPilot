import random
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
from abc import ABC, abstractmethod

class BaseSampler(ABC):
    """Abstract base class for all samplers."""
    
    @abstractmethod
    def sample(self, data: List[Dict[str, Any]], n: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Sample n items from data.
        """
        pass

    def _get_text(self, item: Dict[str, Any]) -> str:
        """Helper to extract text from item for analysis."""
        # Try common keys
        for key in ['input', 'question', 'text', 'source']:
            if key in item:
                return str(item[key])
        return str(item)

    def _get_label(self, item: Dict[str, Any]) -> str:
        """Helper to extract label."""
        for key in ['output', 'label', 'target', 'answer']:
            if key in item:
                return str(item[key])
        return "unknown"

class RandomSampler(BaseSampler):
    """Randomly samples n items."""
    def sample(self, data: List[Dict[str, Any]], n: int, **kwargs) -> List[Dict[str, Any]]:
        if len(data) <= n:
            return data
        return random.sample(data, n)

class HeadSampler(BaseSampler):
    """Samples the first n items."""
    def sample(self, data: List[Dict[str, Any]], n: int, **kwargs) -> List[Dict[str, Any]]:
        return data[:n]

class TailSampler(BaseSampler):
    """Samples the last n items."""
    def sample(self, data: List[Dict[str, Any]], n: int, **kwargs) -> List[Dict[str, Any]]:
        return data[-n:]

class StratifiedSampler(BaseSampler):
    """Samples n items preserving class distribution."""
    def sample(self, data: List[Dict[str, Any]], n: int, **kwargs) -> List[Dict[str, Any]]:
        if len(data) <= n:
            return data
            
        label_map = defaultdict(list)
        for item in data:
            label_map[self._get_label(item)].append(item)
            
        result = []
        # Calculate proportion
        total = len(data)
        for label, items in label_map.items():
            count = int(len(items) / total * n)
            if count > 0:
                result.extend(random.sample(items, count))
        
        # Fill remaining if rounding errors
        remaining = n - len(result)
        if remaining > 0:
            leftover = [x for x in data if x not in result]
            if leftover:
                result.extend(random.sample(leftover, min(remaining, len(leftover))))
                
        return result[:n]

class LengthBalancedSampler(BaseSampler):
    """Samples items from short, medium, and long buckets."""
    def sample(self, data: List[Dict[str, Any]], n: int, **kwargs) -> List[Dict[str, Any]]:
        if len(data) <= n:
            return data
            
        sorted_data = sorted(data, key=lambda x: len(self._get_text(x)))
        third = len(data) // 3
        
        short = sorted_data[:third]
        medium = sorted_data[third:2*third]
        long = sorted_data[2*third:]
        
        per_bucket = n // 3
        result = []
        result.extend(random.sample(short, min(len(short), per_bucket)))
        result.extend(random.sample(medium, min(len(medium), per_bucket)))
        result.extend(random.sample(long, min(len(long), per_bucket)))
        
        # Fill rest randomly
        remaining = n - len(result)
        if remaining > 0:
            leftover = [x for x in data if x not in result]
            result.extend(random.sample(leftover, min(remaining, len(leftover))))
            
        return result

class MaxLengthSampler(BaseSampler):
    """Samples the longest items."""
    def sample(self, data: List[Dict[str, Any]], n: int, **kwargs) -> List[Dict[str, Any]]:
        sorted_data = sorted(data, key=lambda x: len(self._get_text(x)), reverse=True)
        return sorted_data[:n]

class MinLengthSampler(BaseSampler):
    """Samples the shortest items."""
    def sample(self, data: List[Dict[str, Any]], n: int, **kwargs) -> List[Dict[str, Any]]:
        sorted_data = sorted(data, key=lambda x: len(self._get_text(x)))
        return sorted_data[:n]

class KeywordSampler(BaseSampler):
    """Samples items containing specific keywords."""
    def sample(self, data: List[Dict[str, Any]], n: int, keyword: str = "?", **kwargs) -> List[Dict[str, Any]]:
        filtered = [x for x in data if keyword in self._get_text(x)]
        if len(filtered) < n:
            # Fill with random others
            others = [x for x in data if x not in filtered]
            return filtered + random.sample(others, min(n - len(filtered), len(others)))
        return random.sample(filtered, n)

class FewShotSampler(BaseSampler):
    """Samples K items per class (N-way K-shot)."""
    def sample(self, data: List[Dict[str, Any]], n: int, k_shot: int = 5, **kwargs) -> List[Dict[str, Any]]:
        label_map = defaultdict(list)
        for item in data:
            label_map[self._get_label(item)].append(item)
            
        result = []
        for label, items in label_map.items():
            result.extend(random.sample(items, min(len(items), k_shot)))
            
        # If result > n, truncate random
        if len(result) > n:
            return random.sample(result, n)
        return result

class KMeansSampler(BaseSampler):
    """Uses TF-IDF + KMeans to sample representative items."""
    def sample(self, data: List[Dict[str, Any]], n: int, **kwargs) -> List[Dict[str, Any]]:
        if len(data) <= n:
            return data
            
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            from sklearn.metrics import pairwise_distances_argmin_min
            
            texts = [self._get_text(x) for x in data]
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = vectorizer.fit_transform(texts)
            
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            kmeans.fit(X)
            
            # Find closest to centers
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
            
            result = [data[i] for i in closest]
            # If duplicates (rare but possible), fill random
            result = list({id(x): x for x in result}.values()) # dedup by object id
            
            if len(result) < n:
                remaining = n - len(result)
                leftover = [x for x in data if x not in result]
                result.extend(random.sample(leftover, min(remaining, len(leftover))))
                
            return result
            
        except ImportError:
            print("Warning: sklearn not found, falling back to RandomSampler.")
            return RandomSampler().sample(data, n)

class SamplerFactory:
    @staticmethod
    def get_sampler(name: str) -> BaseSampler:
        samplers = {
            "random": RandomSampler(),
            "head": HeadSampler(),
            "tail": TailSampler(),
            "stratified": StratifiedSampler(),
            "length_balanced": LengthBalancedSampler(),
            "max_length": MaxLengthSampler(),
            "min_length": MinLengthSampler(),
            "keyword": KeywordSampler(),
            "few_shot": FewShotSampler(),
            "kmeans": KMeansSampler()
        }
        return samplers.get(name.lower(), RandomSampler())
    
    @staticmethod
    def list_samplers() -> List[str]:
        return [
            "random", "head", "tail", "stratified", 
            "length_balanced", "max_length", "min_length", 
            "keyword", "few_shot", "kmeans"
        ]
