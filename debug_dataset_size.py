
from torchgeo.datasets import EuroSAT
import os

try:
    print("Checking split='train'...")
    ds_train = EuroSAT(root='./data/EuroSAT', split='train', download=False)
    print(f"Size of split='train': {len(ds_train)}")
except Exception as e:
    print(f"split='train' failed: {e}")

try:
    print("Checking split='test'...")
    ds_test = EuroSAT(root='./data/EuroSAT', split='test', download=False)
    print(f"Size of split='test': {len(ds_test)}")
except Exception as e:
    print(f"split='test' failed: {e}")

try:
    print("Checking split='val'...")
    ds_val = EuroSAT(root='./data/EuroSAT', split='val', download=False)
    print(f"Size of split='val': {len(ds_val)}")
except Exception as e:
    print(f"split='val' failed: {e}")
    
try:
    print("Checking split='all' (if supported)...")
    # TorchGeo EuroSAT might not support 'all', but 'train'/'val'/'test' implies a preset split.
    # If we want all, maybe we don't pass split? Or pass something else?
    # Let's try to see if we can instantiate without split or catch error
    pass
except Exception as e:
    print(f"split='all' failed: {e}")
