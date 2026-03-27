from sklearn.model_selection import train_test_split
import torch
import numpy as np

class EuroSATSplitter:
    """
    Splits EuroSAT dataset into Train/Val/Test using scikit-learn.
    """
    def __init__(self, dataset, test_size=0.2, val_size=0.1, random_state=42):
        """
        Args:
            dataset: The full EuroSAT dataset.
            test_size (float): Fraction for test set.
            val_size (float): Fraction of *training* set to use for validation (or absolute fraction of total).
                              Here we treat test_size as first split, then val from remainder?
                              Let's do: Total -> Test (test_size) + Remainder. Remainder -> Val (val_size of total?)
                              Standard: Train/Val/Test ratios.
        """
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        targets = []
        
        # EuroSAT targets are usually single integer labels (0-9).
        # Try to find targets for stratified splitting
        if hasattr(dataset, 'targets'):
             targets = dataset.targets
        elif isinstance(dataset, torch.utils.data.Subset) and hasattr(dataset.dataset, 'targets'):
             targets = [dataset.dataset.targets[i] for i in dataset.indices]
        else:
             # Fallback: lightweight scan if possible, or just random split
             print("Warning: Dataset targets not pre-loaded. Using random split instead of stratified.")
             targets = None

        # 1. Split Test
        if targets is not None:
            train_idx, test_idx = train_test_split(
                self.indices, test_size=test_size, stratify=targets, random_state=random_state
            )
            # Update targets for next split
            train_targets = [targets[i] for i in train_idx]
            
            # 2. Split Val from Train
            # val_size is fraction of original? Let's assume input is e.g. 0.2, 0.2 -> 60/20/20
            # relative val size: val_size / (1 - test_size)
            relative_val_size = val_size / (1 - test_size)
            
            real_train_idx, val_idx = train_test_split(
                train_idx, test_size=relative_val_size, stratify=train_targets, random_state=random_state
            )
        else:
            train_idx, test_idx = train_test_split(
                self.indices, test_size=test_size, random_state=random_state
            )
            relative_val_size = val_size / (1 - test_size)
            real_train_idx, val_idx = train_test_split(
                train_idx, test_size=relative_val_size, random_state=random_state
            )

        self.train_indices = real_train_idx
        self.val_indices = val_idx
        self.test_indices = test_idx

    def get_loaders(self, batch_size=32, num_workers=2, transform=None):
        # We can implement subsets with transforms here if needed.
        # But commonly we just use SubsetRandomSampler or Subset.
        
        train_sub = torch.utils.data.Subset(self.dataset, self.train_indices)
        val_sub = torch.utils.data.Subset(self.dataset, self.val_indices)
        test_sub = torch.utils.data.Subset(self.dataset, self.test_indices)
        
        # Note: If transform (SpectralSelector) needs to be different (aug for train, none for val), 
        # we should wrap subsets. For now, dataset has global transform.
        
        train_loader = torch.utils.data.DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader
