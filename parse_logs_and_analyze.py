import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def parse_log_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex for Epoch line
    # Old: Epoch 1/10 - Loss: 1.7103 - Accuracy: 0.6142 - Time: 55.22s
    # New: Epoch 1/1 - Loss: 2.8285 - Accuracy: 0.1111 - F1: 0.1500 - AUC: 0.8500 - Time: 5.87s
    pattern = re.compile(r"Epoch (\d+)/\d+ - Loss: ([\d.]+) - Accuracy: ([\d.]+)(?: - F1: ([\d.]+))?(?: - AUC: ([\d.]+))? - Time: ([\d.]+)s")
    
    epochs = []
    losses = []
    accuracies = []
    f1s = []
    aucs = []
    times = []

    for match in pattern.finditer(content):
        epochs.append(int(match.group(1)))
        losses.append(float(match.group(2)))
        accuracies.append(float(match.group(3)))
        if match.group(4):
            f1s.append(float(match.group(4)))
        if match.group(5):
            aucs.append(float(match.group(5)))
        times.append(float(match.group(6)))

    return {
        'epochs': epochs,
        'loss': losses,
        'accuracy': accuracies,
        'f1': f1s,
        'auc': aucs,
        'time': times
    }

def analyze_logs(log_dir):
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return

    all_results = []
    
    print(f"Analyzing logs in: {log_dir}")
    
    # Iterate over all output txt files
    for file in log_path.glob('*_output.txt'):
        config_name = file.stem.replace('_output', '')
        data = parse_log_file(file)
        
        if not data['epochs']:
            print(f"Skipping empty or invalid file: {file.name}")
            continue

        # Get best metrics
        best_acc = max(data['accuracy'])
        final_acc = data['accuracy'][-1]
        final_loss = data['loss'][-1]
        best_f1 = max(data['f1']) if data['f1'] else None
        best_auc = max(data['auc']) if data['auc'] else None
        avg_time = sum(data['time']) / len(data['time'])
        
        all_results.append({
            'Configuration': config_name,
            'Best Accuracy': best_acc,
            'Final Accuracy': final_acc,
            'Best F1': best_f1,
            'Best AUC': best_auc,
            'Final Loss': final_loss,
            'Avg Epoch Time (s)': avg_time,
            'data': data
        })

    if not all_results:
        print("No valid results found.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)
    # Sort by Best Accuracy
    df = df.sort_values(by='Best Accuracy', ascending=False)

    print("\nSummary Table:")
    cols_to_show = ['Configuration', 'Best Accuracy', 'Best F1', 'Best AUC', 'Final Accuracy', 'Final Loss', 'Avg Epoch Time (s)']
    print(df[cols_to_show].to_string(index=False))
    
    # Save CSV
    df.drop(columns=['data']).to_csv(log_path / 'summary_metrics.csv', index=False)
    
    # Save Markdown
    markdown_table = df[cols_to_show].to_markdown(index=False)
    with open(log_path / 'summary_table.md', 'w') as f:
        f.write("# Quantum Advantage Search Results\n\n")
        f.write(markdown_table)

    # Plot Convergence (Loss)
    plt.figure(figsize=(12, 8))
    for res in all_results:
        plt.plot(res['data']['epochs'], res['data']['loss'], label=res['Configuration'], marker='o', markersize=3)
    
    plt.title('Training Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(log_path / 'convergence_loss.png')
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(12, 8))
    for res in all_results:
        plt.plot(res['data']['epochs'], res['data']['accuracy'], label=res['Configuration'], marker='o', markersize=3)
    
    plt.title('Validation Accuracy Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(log_path / 'convergence_accuracy.png')
    plt.close()
    
    # Plot F1 if available
    has_f1 = any(res['data']['f1'] for res in all_results)
    if has_f1:
        plt.figure(figsize=(12, 8))
        for res in all_results:
            if res['data']['f1']:
                plt.plot(res['data']['epochs'], res['data']['f1'], label=res['Configuration'], marker='o', markersize=3)
        plt.title('Validation F1-Macro Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(log_path / 'convergence_f1.png')
        plt.close()

    # Plot AUC if available
    has_auc = any(res['data']['auc'] for res in all_results)
    if has_auc:
        plt.figure(figsize=(12, 8))
        for res in all_results:
            if res['data']['auc']:
                plt.plot(res['data']['epochs'], res['data']['auc'], label=res['Configuration'], marker='o', markersize=3)
        plt.title('Validation ROC-AUC Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('ROC-AUC Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(log_path / 'convergence_auc.png')
        plt.close()
    
    print(f"\nPlots saved to {log_path}")

if __name__ == "__main__":
    # Default to the specific log dir requested
    target_dir = "logs/search_advantage_20260126_124604"
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    
    analyze_logs(target_dir)
