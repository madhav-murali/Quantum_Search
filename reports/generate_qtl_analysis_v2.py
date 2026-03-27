#!/usr/bin/env python3
"""
IMPROVED QTL Analysis with Accurate Parameter Counting

Generates comprehensive analysis with:
- Accurate parameter counts from actual models
- Training convergence plots
- Accuracy comparison
- Parameter efficiency analysis
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import torch
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.backbones import BackboneFactory
from src.models.hybrid_model import HybridGeoModel

# Configuration with ACCURATE parameter counts
def get_actual_param_count(config_name, n_qubits=8, q_depth=3):
    """Calculate actual parameter count by building the model."""
    from run_experiments import CONFIGS
    
    if config_name not in CONFIGS:
        return None
    
    config = CONFIGS[config_name]
    
    try:
        # Build model
        backbone, feature_dim = BackboneFactory.create(
            config['backbone'], 
            pretrained=False,
            in_channels=3
        )
        
        if config.get('quantum', False):
            model = HybridGeoModel(
                backbone=backbone,
                feature_dim=feature_dim,
                n_classes=10,
                n_qubits=n_qubits,
                n_qlayers=q_depth,
                encoding=config.get('encoding', 'angle'),
                ansatz=config.get('ansatz', 'vqc'),
                q_type=config.get('q_type', 'standard'),
                standard_dim=config.get('standard_dim', None),
                freeze_quantum=config.get('freeze_quantum', False)
            )
        else:
            # Classical model
            import torch.nn as nn
            if 'resnet' in config['backbone']:
                model = backbone
                model.fc = nn.Linear(feature_dim, 10)
            else:
                model = nn.Sequential(backbone, nn.Linear(feature_dim, 10))
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
    except Exception as e:
        print(f"Warning: Could not calculate params for {config_name}: {e}")
        return None

CONFIGS_INFO = {
    'qtl_source_resnet_angle': {
        'name': 'ResNet50 (angle)',
        'color': '#e74c3c',
        'marker': 'p',
        'group': 'source'
    },
    'qtl_lenet_frozen_angle': {
        'name': 'LeNet5 QTL Frozen',
        'color': '#2ecc71',
        'marker': '^',
        'group': 'qtl'
    },
    'qtl_lenet_finetuned_angle': {
        'name': 'LeNet5 QTL Fine-tuned',
        'color': '#27ae60',
        'marker': 'D',
        'group': 'qtl'
    },
    'lenet_quantum_angle_vqc': {
        'name': 'LeNet5 Baseline (angle)',
        'color': '#95a5a6',
        'marker': 'o',
        'group': 'baseline'
    },
    'lenet5_quantum_amplitude_vqc': {
        'name': 'LeNet5 Original (amplitude)',
        'color': '#3498db',
        'marker': 's',
        'group': 'baseline'
    },
    'hybrid_resnet_angle_vqc': {
        'name': 'ResNet50 Hybrid (reference)',
        'color': '#9b59b6',
        'marker': 'h',
        'group': 'reference'
    },
    'qtl_lenet_frozen': {
        'name': 'LeNet5 QTL Frozen (bad)',
        'color': '#e67e22',
        'marker': 'v',
        'group': 'failed'
    },
    'qtl_lenet_finetuned': {
        'name': 'LeNet5 QTL Tuned (bad)',
        'color': '#d35400',
        'marker': '<',
        'group': 'failed'
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
    for config_name, config_info in CONFIGS_INFO.items():
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
                markersize=5,
                markevery=max(1, len(epochs)//10),
                alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Convergence: Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([20, 100])
    
    # Plot 2: Loss over epochs
    for config_name, config_info in CONFIGS_INFO.items():
        data = results_data.get(config_name)
        if data is None:
            continue
        
        epochs = range(1, len(data['loss']) + 1)
        
        ax2.plot(epochs, data['loss'],
                marker=config_info['marker'],
                color=config_info['color'],
                label=config_info['name'],
                linewidth=2,
                markersize=5,
                markevery=max(1, len(epochs)//10),
                alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Training Convergence: Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/qtl_convergence_comparison.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: results/qtl_convergence_comparison.png')
    plt.close()

def create_accuracy_comparison(results_data, param_counts):
    """Create accuracy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Prepare data - group by category
    groups = {'source': [], 'qtl': [], 'baseline': [], 'reference': [], 'failed': []}
    
    for config_name, config_info in CONFIGS_INFO.items():
        data = results_data.get(config_name)
        if data is None:
            continue
        
        group = config_info['group']
        groups[group].append({
            'name': config_info['name'],
            'best_acc': max(data['val_acc']) * 100,
            'final_acc': data['val_acc'][-1] * 100,
            'color': config_info['color']
        })
    
    # Plot grouped bars
    x_pos = 0
    x_labels = []
    x_ticks = []
    
    for group_name, items in groups.items():
        if not items:
            continue
        
        for item in items:
            ax.bar(x_pos, item['best_acc'], 
                   color=item['color'], alpha=0.8, 
                   edgecolor='black', linewidth=1.5,
                   label=item['name'])
            
            # Add value label
            ax.text(x_pos, item['best_acc'] + 1,
                   f'{item["best_acc"]:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            x_labels.append(item['name'])
            x_ticks.append(x_pos)
            x_pos += 1
        
        x_pos += 0.5  # Gap between groups
    
    ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('QTL Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=15, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Add horizontal line at 90% (target)
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target (90%)')
    
    plt.tight_layout()
    plt.savefig('results/qtl_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: results/qtl_accuracy_comparison.png')
    plt.close()

def create_parameter_efficiency_plot(results_data, param_counts):
    """Create parameter efficiency scatter plot with ACCURATE counts."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for config_name, config_info in CONFIGS_INFO.items():
        data = results_data.get(config_name)
        params = param_counts.get(config_name)
        
        if data is None or params is None:
            continue
        
        accuracy = max(data['val_acc']) * 100
        total_params = params['total']
        
        # Plot point
        ax.scatter(total_params, accuracy, 
                  s=400,
                  c=config_info['color'],
                  marker=config_info['marker'],
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=2,
                  label=config_info['name'],
                  zorder=3)
        
        # Add annotation
        param_str = f'{total_params/1e6:.1f}M' if total_params > 1e6 else f'{total_params/1e3:.0f}K'
        ax.annotate(f'{accuracy:.1f}%\\n{param_str}',
                   (total_params, accuracy),
                   textcoords="offset points",
                   xytext=(0, 15),
                   ha='center',
                   fontsize=8,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=config_info['color'], alpha=0.3))
    
    ax.set_xlabel('Total Parameters (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Efficiency: Accuracy vs Model Size', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_xscale('log')
    ax.set_ylim([20, 100])
    
    # Add target line
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/qtl_parameter_efficiency.png', dpi=300, bbox_inches='tight')
    print('✓ Saved: results/qtl_parameter_efficiency.png')
    plt.close()

def create_detailed_table(results_data, param_counts):
    """Create detailed results table."""
    table_data = []
    
    for config_name, config_info in CONFIGS_INFO.items():
        data = results_data.get(config_name)
        params = param_counts.get(config_name)
        
        if data is None:
            continue
        
        best_acc = max(data['val_acc']) * 100
        final_acc = data['val_acc'][-1] * 100
        best_f1 = max(data['val_f1']) * 100
        final_loss = data['loss'][-1]
        avg_epoch_time = np.mean(data['epoch_times'])
        total_time = sum(data['epoch_times']) / 60  # minutes
        
        if params:
            param_str = f'{params["total"]/1e6:.2f}M' if params["total"] > 1e6 else f'{params["total"]/1e3:.1f}K'
            trainable_str = f'{params["trainable"]/1e6:.2f}M' if params["trainable"] > 1e6 else f'{params["trainable"]/1e3:.1f}K'
        else:
            param_str = 'N/A'
            trainable_str = 'N/A'
        
        table_data.append({
            'Model': config_info['name'],
            'Parameters': param_str,
            'Trainable': trainable_str,
            'Best Acc (%)': f'{best_acc:.2f}',
            'Final Acc (%)': f'{final_acc:.2f}',
            'Best F1 (%)': f'{best_f1:.2f}',
            'Final Loss': f'{final_loss:.4f}',
            'Avg Epoch (s)': f'{avg_epoch_time:.1f}',
            'Total Time (min)': f'{total_time:.1f}',
            'Group': config_info['group']
        })
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    df.to_csv('results/qtl_detailed_results.csv', index=False)
    print('✓ Saved: results/qtl_detailed_results.csv')
    
    # Save as markdown
    with open('results/qtl_detailed_results.md', 'w') as f:
        f.write('# Quantum Transfer Learning - Detailed Results\n\n')
        f.write(df.to_markdown(index=False))
        
        # Analysis
        f.write('\n\n## Key Findings\n\n')
        
        # Find best QTL and baseline
        qtl_rows = df[df['Group'] == 'qtl']
        baseline_rows = df[df['Group'] == 'baseline']
        
        if not qtl_rows.empty:
            best_qtl = qtl_rows.loc[qtl_rows['Best Acc (%)'].astype(float).idxmax()]
            f.write(f'**Best QTL:** {best_qtl["Model"]} - {best_qtl["Best Acc (%)"]}%\n\n')
        
        if not baseline_rows.empty:
            best_baseline = baseline_rows.loc[baseline_rows['Best Acc (%)'].astype(float).idxmax()]
            f.write(f'**Best Baseline:** {best_baseline["Model"]} - {best_baseline["Best Acc (%)"]}%\n\n')
        
        if not qtl_rows.empty and not baseline_rows.empty:
            benefit = float(best_qtl['Best Acc (%)']) - float(best_baseline['Best Acc (%)'])
            f.write(f'**Transfer Benefit:** {benefit:+.2f}%\n\n')
    
    print('✓ Saved: results/qtl_detailed_results.md')
    
    return df

def print_summary(results_data, param_counts):
    """Print summary to console."""
    print('\n' + '='*90)
    print('QUANTUM TRANSFER LEARNING - RESULTS SUMMARY')
    print('='*90)
    print(f'{"Model":<40} {"Params":<12} {"Best Acc":<12} {"Status":<20}')
    print('-'*90)
    
    for config_name, config_info in CONFIGS_INFO.items():
        data = results_data.get(config_name)
        params = param_counts.get(config_name)
        
        if data is None:
            print(f'{config_info["name"]:<40} {"N/A":<12} {"-":<12} {"❌ Not run":<20}')
            continue
        
        best_acc = max(data['val_acc']) * 100
        param_str = f'{params["total"]/1e6:.1f}M' if params and params["total"] > 1e6 else f'{params["total"]/1e3:.0f}K' if params else 'N/A'
        
        # Status
        if best_acc >= 90:
            status = '✅ Excellent (≥90%)'
        elif best_acc >= 80:
            status = '⚠️  Good (80-90%)'
        elif best_acc >= 70:
            status = '⚠️  Fair (70-80%)'
        else:
            status = '❌ Poor (<70%)'
        
        print(f'{config_info["name"]:<40} {param_str:<12} {best_acc:>6.2f}%     {status:<20}')
    
    print('='*90 + '\n')

def main():
    """Main analysis function."""
    print('\n' + '='*90)
    print('Generating IMPROVED QTL Analysis with Accurate Parameter Counting')
    print('='*90 + '\n')
    
    # Calculate actual parameter counts
    print('Calculating accurate parameter counts...\n')
    param_counts = {}
    for config_name in CONFIGS_INFO.keys():
        params = get_actual_param_count(config_name, n_qubits=8, q_depth=3)
        if params:
            param_counts[config_name] = params
            param_str = f'{params["total"]/1e6:.2f}M' if params["total"] > 1e6 else f'{params["total"]/1e3:.0f}K'
            print(f'  {config_name:<35} {param_str:>10} ({params["trainable"]/1e3:.0f}K trainable)')
    
    print(f'\n✓ Calculated {len(param_counts)} parameter counts\n')
    
    # Load all results
    results_data = {}
    for config_name in CONFIGS_INFO.keys():
        data = load_results(config_name)
        if data:
            results_data[config_name] = data
            print(f'✓ Loaded: {config_name}')
        else:
            print(f'⚠  Missing: {config_name}')
    
    if not results_data:
        print('\n❌ No results found! Run experiments first.')
        return
    
    print(f'\nFound {len(results_data)} result files\n')
    
    # Create visualizations
    print('Creating visualizations...\n')
    create_convergence_plot(results_data)
    create_accuracy_comparison(results_data, param_counts)
    create_parameter_efficiency_plot(results_data, param_counts)
    
    # Create tables
    print('\nCreating result tables...\n')
    df = create_detailed_table(results_data, param_counts)
    
    # Print summary
    print_summary(results_data, param_counts)
    
    print('='*90)
    print('✓ Analysis complete!')
    print('='*90)
    print('\nGenerated files:')
    print('  - results/qtl_convergence_comparison.png')
    print('  - results/qtl_accuracy_comparison.png')
    print('  - results/qtl_parameter_efficiency.png')
    print('  - results/qtl_detailed_results.csv')
    print('  - results/qtl_detailed_results.md')
    print('\n')

if __name__ == '__main__':
    main()
