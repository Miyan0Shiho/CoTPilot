import os
import sys

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "opencompass"))
sys.path.append(os.path.join(project_root, "EvoPrompt"))
sys.path.append(current_dir)

from cot_pilot.core.dataset_manager import DatasetManager
from cot_pilot.core.sampler import SamplerFactory

def test():
    print("Testing DatasetManager...")
    dm = DatasetManager()
    datasets = dm.list_datasets()
    print(f"Found {len(datasets)} datasets.")
    if len(datasets) > 0:
        print(f"First 5: {datasets[:5]}")
        
    # Test Sampler
    print("\nTesting Sampler...")
    data = [{"text": f"item {i}", "label": i%2} for i in range(100)]
    factory = SamplerFactory()
    
    for strategy in factory.list_samplers():
        sampler = factory.get_sampler(strategy)
        try:
            sample = sampler.sample(data, 5)
            print(f"Strategy {strategy}: {len(sample)} items")
        except Exception as e:
            print(f"Strategy {strategy} failed: {e}")

if __name__ == "__main__":
    test()
