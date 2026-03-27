#!/usr/bin/env python3
import os
import glob
import json
import matplotlib.pyplot as plt

def main():
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print("No results directory found.")
        return
        
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    if not json_files:
        print("No json result files found.")
        return
        
    latest_file = max(json_files, key=os.path.getmtime)
    print(f"Loading latest run: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
        
    if 'loss' in data:
        plot_single_run(data, os.path.basename(latest_file))
    elif 'strategies' in data:
        print("This is a summary file. Plotting might not be fully supported without epoch data.")
    else:
        # Check if it's a nested specific strategy file
        if any(isinstance(v, dict) and 'loss' in v for v in data.values()):
             for k, v in data.items():
                 if isinstance(v, dict) and 'loss' in v:
                     plot_single_run(v, f"{os.path.basename(latest_file)} - {k}")
        else:
             plot_single_run(data, os.path.basename(latest_file))

def plot_single_run(data, title):
    epochs = range(1, len(data.get('loss', [])) + 1)
    if not epochs:
        print(f"Skipping {title} (no loss data found).")
        return
        
    has_auc = 'val_roc_auc' in data or 'roc_auc' in data
    ncols = 3 if has_auc else 2
    plt.figure(figsize=(5 * ncols, 5))
    
    # Plot Loss
    plt.subplot(1, ncols, 1)
    plt.plot(epochs, data.get('loss', []), label='Train Loss', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot Acc / F1
    plt.subplot(1, ncols, 2)
    plt.plot(epochs, data.get('val_acc', []), label='Val Acc', marker='x')
    plt.plot(epochs, data.get('val_f1', []), label='Val F1', marker='s')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot ROC AUC if present
    if has_auc:
        auc_key = 'val_roc_auc' if 'val_roc_auc' in data else 'roc_auc'
        if isinstance(data[auc_key], list):
             plt.subplot(1, ncols, 3)
             plt.plot(epochs, data[auc_key], label='ROC AUC', marker='^', color='green')
             plt.title('ROC AUC')
             plt.xlabel('Epoch')
             plt.legend()
             plt.grid(True)

    plt.suptitle(f"Run: {title}")
    plt.tight_layout()
    plot_path = "latest_run_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
