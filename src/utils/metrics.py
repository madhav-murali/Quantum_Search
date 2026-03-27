import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate Accuracy, F1 scores, and optionally ROC-AUC.
    Args:
        y_true (np.array): Ground truth (N,)
        y_pred (np.array): Predictions (N,)
        y_prob (np.array): Probabilities (N, C)
    Returns:
        dict: Metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }
    
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            metrics["roc_auc"] = auc
        except ValueError:
            pass # Fails if not all classes are present in y_true, etc.
            
    return metrics
