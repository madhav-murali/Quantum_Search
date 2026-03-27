import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def analyze_results():
    results_path = Path('results')
    if not results_path.exists():
        print("No results found.")
        return

    all_data = []
    
    # 1. Load Data
    for file in results_path.glob('*_results.json'):
        config_name = file.stem.replace('_results', '')
        with open(file, 'r') as f:
            data = json.load(f)
            
        # Summary metrics (last epoch)
        final_loss = data['loss'][-1]
        final_acc = data.get('val_acc', data.get('val_mAP', [0]))[-1]
        final_f1 = data['val_f1'][-1]
        
        all_data.append({
            'Configuration': config_name,
            'Validation Accuracy': final_acc,
            'Validation F1': final_f1,
            'Loss': final_loss
        })
        
        # 2. Convergence Plot
        plt.figure()
        plt.plot(data['loss'], label='Training Loss')
        plt.title(f'Convergence: {config_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(results_path / f'{config_name}_convergence.png')
        plt.close()

    # 3. Create DataFrame
    df = pd.DataFrame(all_data)
    print("\nSummary Table:")
    print(df)
    
    # Save Table
    df.to_csv(results_path / 'summary_table.csv', index=False)
    
    # 4. Bar Chart (Accuracy/mAP)
    if not df.empty:
        plt.figure(figsize=(10, 6))
        # Use 'Validation Accuracy' as it replaced 'Validation mAP'
        metric = 'Validation Accuracy'
        if metric in df.columns:
            plt.bar(df['Configuration'], df[metric])
            plt.title(f'{metric} by Configuration')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(results_path / 'comparison_bar_chart.png')
            plt.close()
    
    # 5. Markdown Table Generation
    markdown_table = df.to_markdown(index=False)
    with open(results_path / 'summary_table.md', 'w') as f:
        f.write(markdown_table)
        
    print(f"\nAnalysis complete. Results saved to {results_path}")

if __name__ == '__main__':
    analyze_results()
