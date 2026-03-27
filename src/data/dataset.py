import torch
from torchgeo.datasets import EuroSAT
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import Subset
import torch.nn as nn
import os
import numpy as np

class SpectralSelector(nn.Module):
    """
    Selects specific spectral bands from the input tensor.
    
    Args:
        mode (str): 'RGB' for visible bands or 'ALL' for all 13 bands.
    """
    def __init__(self, mode='RGB'):
        super().__init__()
        self.mode = mode.upper()
        
        # EuroSAT (Sentinel-2) indices: 0 to 12.
        # RGB = B04 (3), B03 (2), B02 (1)
        
        if self.mode == 'RGB':
            self.indices = [3, 2, 1]
        else:
            self.indices = list(range(13))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (C, H, W) or (B, C, H, W)
        Returns:
            Tensor: Selected bands.
        """
        if isinstance(x, dict):
            img = x['image']
            x['image'] = img[..., self.indices, :, :]
            return x
        else:
             return x[..., self.indices, :, :]

    def __repr__(self):
        return f"SpectralSelector(mode={self.mode})"

from torchvision.transforms import Resize

class DictResize(nn.Module):
    """
    Applies Resize to the 'image' key in the sample dictionary.
    """
    def __init__(self, size):
        super().__init__()
        self.resize = Resize(size)
        
    def forward(self, x):
        if isinstance(x, dict):
            x['image'] = self.resize(x['image'])
            return x
        return self.resize(x)


def get_dataset(root, dataset_name="EuroSAT", download=True, subset_fraction=1.0):
    """
    Get dataset by name and apply subset fraction if requested.
    """
    if dataset_name.lower() == "eurosat":
        ds = EuroSAT(
            root=root,
            split='train',
            transforms=None,
            download=download,
            checksum=False
        )
    else:
        # Fallback to standard ImageFolder for datasets like SIRI-WHU / UC_M_LUC
        # Assuming the root directly points to the dataset folder or we append dataset_name
        path = os.path.join(root, dataset_name) if os.path.basename(root) != dataset_name else root
        img_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor()
        ])
        ds = ImageFolder(root=path, transform=img_transform)
        
    if subset_fraction < 1.0:
        total = len(ds)
        indices = np.random.choice(total, int(total * subset_fraction), replace=False)
        ds = Subset(ds, indices)
    
    return ds

def create_dataloader(dataset, batch_sampler=None, batch_size=32, shuffle=True, num_workers=2):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1 if batch_sampler else batch_size, # If batch_sampler is used, batch_size is ignored
        batch_sampler=batch_sampler,
        shuffle=shuffle if batch_sampler is None else False,
        num_workers=num_workers,
        collate_fn=None # Default
    )
