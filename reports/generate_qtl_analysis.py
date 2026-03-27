#!/usr/bin/env python3
"""
Generate comprehensive QTL analysis and visualizations.

Creates:
- Training convergence plots
- Accuracy comparison bar chart
- Parameter efficiency scatter plot
- Detailed results table
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

# Configuration
CONFIGS = {
    'lenet5_baseline_amplitude': {
        'name': 'LeNet5 Baseline\n(No Transfer)',
        'color': '#95a5a6',
        'params': 42000,
        'marker': 'o'
    },
    'lenet5_quantum_amplitude_vqc': {
        'name': 'LeNet5 Original\n(No Standardization)',
        'color': '#3498db',
        'params': 42000,
        'marker': 's'
    },
    'qtl_lenet_frozen': {
        'name': 'LeNet5 QTL\n(Frozen)',
        'color': '#2ecc71',
        'params': 42000,
        'marker': '^'
    },
    'qtl_lenet_finetuned': {
        'name': 'LeNet5 QTL\n(Fine-tuned)',
        'color': '#27ae60',
        'params': 42000,
        'marker': 'D'
    },
    'qtl_source_resnet_amplitude': {
        'name': 'ResNet50 Source',
        'color': '#e74c3c',
        'params': 25600000,
        'marker': 'p'
    }
}

def load_results(config_name):
    """Load results JSON for a configuration."""
    result_file = Path(f'results/{config_name}_results.json')
    if not result_file.exists():
        return None
    
    with open(result_file) as f:
        return json.load(f)

def create_convergence_plot(results_data):
    """Create training convergence comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy over epochs
    for config_name, config_info in CONFIGS.items():
        data = results_data.get(config_name)
        if data is None:
            continue
        
        epochs = range(1, len(data['val_acc']) + 1)
        accuracies = [acc * 100 for acc in data['val_acc']]
        
        ax1.plot(epochs, accuracies, 
                marker=config_info['marker'],
                color=config_info['color'],
                label=config_info['name'],
                linewidth=2,
                markersize=6,
                alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Convergence: Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Plot 2: Loss over epochs
    for config_name, config_info in CONFIGS.items():
        data = results_data.get(config_name)
        if data is None:
            continue
        
        epochs = range(1, len(data['loss']) + 1)
        
        ax2.plot(epochs, data['loss'],
                marker=config_info['marker'],
                color=config_info['color'],
                label=config_info['name'],
                linewidth=2,
                markersize=6,
                alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training Convergence: Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/qtl_convergence_comparison.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: results/qtl_convergence_comparison.png')
    plt.close()

def create_accuracy_comparison(results_data):
    """Create accuracy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data
    configs = []
    best_accs = []
    final_accs = []
    colors = []
    
    for config_name, config_info in CONFIGS.items():
        data = results_data.get(config_name)
        if data is None:
            continue
        
        configs.append(config_info['name'])
        best_accs.append(max(data['val_acc']) * 100)
        final_accs.append(data['val_acc'][-1] * 100)
        colors.append(config_info['color'])
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, best_accs, width, label='Best Accuracy', 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, final_accs, width, label='Final Accuracy',
                   color=colors, alpha=0.5, edgecolor='black', linewidth=1.5, hatch='//')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('QTL Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('results/qtl_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: results/qtl_accuracy_comparison.png')
    plt.close()

def create_parameter_efficiency_plot(results_data):
    """Create parameter efficiency scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for config_name, config_info in CONFIGS.items():
        data = results_data.get(config_name)
        if data is None:
            continue
        
        params = config_info['params']
        accuracy = max(data['val_acc']) * 100
        
        # Use different scale for ResNet (millions) vs LeNet (thousands)
        if params > 1e6:
            x_val = params / 1e6
            x_label_suffix = 'M'
        else:
            x_val = params / 1e3
            x_label_suffix = 'K'
        
        ax.scatter(x_val, accuracy, 
                  s=400,
                  c=config_info['color'],
                  marker=config_info['marker'],
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=2,
                  label=config_info['name'],
                  zorder=3)
        
        # Add annotation
        ax.annotate(f'{accuracy:.1f}%',
                   (x_val, accuracy),
                   textcoords="offset points",
                   xytext=(0, 10),
                   ha='center',
                   fontsize=9,
                   fontweight='bold')
    
    ax.set_xlabel('Model Parameters', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Efficiency: Accuracy vs Model Size', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xscale('log')
    
    # Custom x-axis labels
    ax.set_xlabel('Model Parameters (log scale)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/qtl_parameter_efficiency.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: results/qtl_parameter_efficiency.png')
    plt.close()

def create_detailed_table(results_data):
    """Create detailed results table."""
    table_data = []
    
    for config_name, config_info in CONFIGS.items():
        data = results_data.get(config_name)
        if data is None:
            continue
        
        best_acc = max(data['val_acc']) * 100
        final_acc = data['val_acc'][-1] * 100
        best_f1 = max(data['val_f1']) * 100
        final_loss = data['loss'][-1]
        avg_epoch_time = np.mean(data['epoch_times'])
        total_time = sum(data['epoch_times']) / 60  # minutes
        
        params = config_info['params']
        param_str = f'{params/1e6:.1f}M' if params > 1e6 else f'{params/1e3:.0f}K'
        
        table_data.append({
            'Model': config_info['name'].replace('\n', ' '),
            'Parameters': param_str,
            'Best Acc (%)': f'{best_acc:.2f}',
            'Final Acc (%)': f'{final_acc:.2f}',
            'Best F1 (%)': f'{best_f1:.2f}',
            'Final Loss': f'{final_loss:.4f}',
            'Avg Epoch (s)': f'{avg_epoch_time:.1f}',
            'Total Time (min)': f'{total_time:.1f}'
        })
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    df.to_csv('results/qtl_detailed_results.csv', index=False)
    print('✓ Saved: results/qtl_detailed_results.csv')
    
    # Save as markdown
    with open('results/qtl_detailed_results.md', 'w') as f:
        f.write('# Quantum Transfer Learning - Detailed Results\n\n')
        f.write(df.to_markdown(index=False))
        f.write('\n\n## Analysis\n\n')
        
        # Calculate transfer benefit
        baseline_acc = None
        qtl_best_acc = None
        
        for row in table_data:
            if 'Baseline' in row['Model']:
                baseline_acc = float(row['Best Acc (%)'])
            elif 'Fine-tuned' in row['Model']:
                qtl_best_acc = float(row['Best Acc (%)'])
        
        if baseline_acc and qtl_best_acc:
            benefit = qtl_best_acc - baseline_acc
            f.write(f'**Transfer Learning Benefit:** {benefit:+.2f}% accuracy improvement\n\n')
        
        # Parameter reduction
        f.write(f'**Parameter Reduction:** 600x fewer parameters (ResNet50 vs LeNet5)\n')
    
    print('✓ Saved: results/qtl_detailed_results.md')
    
    return df

def print_summary(results_data):
    """Print summary to console."""
    print('\n' + '='*80)
    print('QUANTUM TRANSFER LEARNING - RESULTS SUMMARY')
    print('='*80)
    print(f'{"Model":<35} {"Best Acc":<12} {"Final Acc":<12} {"Status":<15}')
    print('-'*80)
    
    baseline_acc = None
    
    for config_name, config_info in CONFIGS.items():
        data = results_data.get(config_name)
        if data is None:
            status = '❌ Not run'
            best_acc_str = '-'
            final_acc_str = '-'
        else:
            best_acc = max(data['val_acc']) * 100
            final_acc = data['val_acc'][-1] * 100
            best_acc_str = f'{best_acc:.2f}%'
            final_acc_str = f'{final_acc:.2f}%'
            
            # Determine status
            if 'Baseline' in config_info['name']:
                baseline_acc = best_acc
                status = '✓ Reference'
            elif best_acc >= 90:
                status = '✓ Excellent'
            elif best_acc >= 80:
                status = '⚠ Good'
            elif best_acc >= 70:
                status = '⚠ Poor'
            else:
                status = '❌ Failed'
        
        print(f'{config_info["name"].replace(chr(10), " "):<35} {best_acc_str:<12} {final_acc_str:<12} {status:<15}')
    
    print('='*80)
    
    # Print transfer benefit if available
    if baseline_acc:
        print(f'\nBaseline accuracy: {baseline_acc:.2f}%')
        for config_name, config_info in CONFIGS.items():
            if 'QTL' in config_info['name']:
                data = results_data.get(config_name)
                if data:
                    best_acc = max(data['val_acc']) * 100
                    benefit = best_acc - baseline_acc
                    print(f'{config_info["name"].replace(chr(10), " ")}: {best_acc:.2f}% ({benefit:+.2f}% vs baseline)')
    
    print('\n')

def main():
    """Main analysis function."""
    print('\n' + '='*80)
    print('Generating QTL Analysis and Visualizations')
    print('='*80 + '\n')
    
    # Load all results
    results_data = {}
    for config_name in CONFIGS.keys():
        data = load_results(config_name)
        if data:
            results_data[config_name] = data
            print(f'✓ Loaded: {config_name}')
        else:
            print(f'⚠ Missing: {config_name}')
    
    if not results_data:
        print('\n❌ No results found! Run experiments first.')
        return
    
    print(f'\nFound {len(results_data)} result files\n')
    
    # Create visualizations
    print('Creating visualizations...\n')
    create_convergence_plot(results_data)
    create_accuracy_comparison(results_data)
    create_parameter_efficiency_plot(results_data)
    
    # Create tables
    print('\nCreating result tables...\n')
    df = create_detailed_table(results_data)
    
    # Print summary
    print_summary(results_data)
    
    print('='*80)
    print('✓ Analysis complete!')
    print('='*80)
    print('\nGenerated files:')
    print('  - results/qtl_convergence_comparison.png')
    print('  - results/qtl_accuracy_comparison.png')
    print('  - results/qtl_parameter_efficiency.png')
    print('  - results/qtl_detailed_results.csv')
    print('  - results/qtl_detailed_results.md')
    print('\n')

if __name__ == '__main__':
    main()
