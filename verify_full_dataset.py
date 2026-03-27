
from torchgeo.datasets import EuroSAT
from torch.utils.data import ConcatDataset
import ssl
import urllib.request

# Disable SSL verification globally for this script
ssl._create_default_https_context = ssl._create_unverified_context

try:
    print("Loading all splits...")
    # Using download=True to ensure they are prepared if needed (data should be there)
    ds_train = EuroSAT(root='./data/EuroSAT', split='train', download=True)
    ds_val = EuroSAT(root='./data/EuroSAT', split='val', download=True)
    ds_test = EuroSAT(root='./data/EuroSAT', split='test', download=True)
    
    print(f"Train: {len(ds_train)}")
    print(f"Val: {len(ds_val)}")
    print(f"Test: {len(ds_test)}")
    
    full_ds = ConcatDataset([ds_train, ds_val, ds_test])
    print(f"Total Combined: {len(full_ds)}")
    
except Exception as e:
    print(f"Error: {e}")
