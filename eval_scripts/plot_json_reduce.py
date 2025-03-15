import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def load_experiment_results(json_files):
    """
    Load results from multiple experiment JSON files
    
    Args:
        json_files (list): List of paths to JSON result files
    
    Returns:
        dict: Dictionary with experiment names as keys and their results as values
    """
    experiments = {}
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
                
            # Extract experiment name from the filename
            exp_name = Path(json_file).stem.split('_results')[0]
            experiments[exp_name] = results
            print(f"Loaded experiment: {exp_name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return experiments

def compare_against_baseline(experiments, baseline_exp, metric_ga, metric_sml, title, ylabel, output_path, y_max=None):
    """
    Create a comparison plot of different SML methods against a common GA baseline
    
    Args:
        experiments (dict): Dictionary of experiments and their results
        baseline_exp (str): Name of the experiment to use as the baseline
        metric_ga (str): Key for the GA (Global Alignment) metric
        metric_sml (str): Key for the SML (Scale Map Learning) metric
        title (str): Plot title
        ylabel (str): Y-axis label
        output_path (str): Path to save the plot
        y_max (float, optional): Maximum y-axis value
    """
    if baseline_exp not in experiments:
        print(f"Error: Baseline experiment '{baseline_exp}' not found")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Set a colorful palette for SML lines
    palette = sns.color_palette("tab10", len(experiments))
    markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*']
    
    # Get baseline data
    baseline_percentages = experiments[baseline_exp]['reduction_percentages']
    baseline_values = experiments[baseline_exp][metric_ga]
    
    # Plot baseline GA as a dashed black line
    baseline_line, = plt.plot(baseline_percentages, baseline_values, 
                             color='black', 
                             linestyle='--',
                             linewidth=3,
                             marker='o',
                             markersize=8,
                             label=f"Baseline (GA)")
    
    # Plot each experiment's SML results
    sml_lines = []
    for i, (exp_name, results) in enumerate(experiments.items()):
        if exp_name == baseline_exp and len(experiments) > 1:
            # For baseline experiment, use a different color to distinguish it
            line_color = 'darkblue'
            zorder = 10  # Make sure it's on top
        else:
            line_color = palette[i]
            zorder = 5
            
        # Handle potential differences in reduction percentages between experiments
        percentages = results['reduction_percentages']
        
        line, = plt.plot(percentages, results[metric_sml], 
                       color=line_color, 
                       linestyle='-',
                       linewidth=2,
                       marker=markers[i % len(markers)], 
                       markersize=8,
                       zorder=zorder,
                       label=f"{exp_name} (SML)")
        sml_lines.append(line)
    
    plt.xlabel('Point Reduction Percentage (%)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add all lines to the legend
    plt.legend(handles=[baseline_line] + sml_lines, loc='best', fontsize=12)
    
    # Set axis limits if needed
    if y_max is not None:
        plt.ylim(top=y_max)
    
    # Improve tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add x and y axis ticks at specific intervals
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {title} plot to {output_path}")

def create_improvement_table(experiments, baseline_exp, output_path):
    """
    Create a table showing the percentage improvement of each SML method over the baseline GA
    
    Args:
        experiments (dict): Dictionary of experiments and their results
        baseline_exp (str): Name of the experiment to use as the baseline
        output_path (str): Path to save the table
    """
    if baseline_exp not in experiments:
        print(f"Error: Baseline experiment '{baseline_exp}' not found")
        return
    
    # Get baseline data
    baseline_percentages = experiments[baseline_exp]['reduction_percentages']
    
    # Metrics to analyze
    metrics = [
        ("RMSE", "rmse_ga", "rmse_sml"),
        ("MAE", "mae_ga", "mae_sml"),
        ("AbsRel", "absrel_ga", "absrel_sml"),
        ("iRMSE", "inv_rmse_ga", "inv_rmse_sml"),
        ("iMAE", "inv_mae_ga", "inv_mae_sml"),
        ("iAbsRel", "inv_absrel_ga", "inv_absrel_sml")
    ]
    
    with open(output_path, 'w') as f:
        f.write("# Improvement Over Baseline GA\n\n")
        f.write(f"Baseline experiment: **{baseline_exp}**\n\n")
        
        # For each reduction percentage, create a separate table
        for p_idx, p in enumerate(baseline_percentages):
            f.write(f"## At {p}% Point Reduction\n\n")
            
            # Create table header
            header = "| Metric | Baseline GA |"
            for exp_name in experiments:
                header += f" {exp_name} SML | {exp_name} Improvement |"
            f.write(header + "\n")
            
            # Create table separator
            separator = "|--------|------------|"
            for _ in experiments:
                separator += "----------|-----------|"
            f.write(separator + "\n")
            
            # Add rows for each metric
            for metric_name, ga_key, sml_key in metrics:
                baseline_value = experiments[baseline_exp][ga_key][p_idx]
                
                row = f"| {metric_name} | {baseline_value:.3f} |"
                
                for exp_name, results in experiments.items():
                    # Get the SML value for this experiment
                    if p in results['reduction_percentages']:
                        exp_p_idx = results['reduction_percentages'].index(p)
                        sml_value = results[sml_key][exp_p_idx]
                    else:
                        # Find closest percentage if exact match not available
                        closest_p = min(results['reduction_percentages'], key=lambda x: abs(x - p))
                        exp_p_idx = results['reduction_percentages'].index(closest_p)
                        sml_value = results[sml_key][exp_p_idx]
                    
                    # Calculate percentage improvement
                    # For these metrics, lower is better
                    improvement = ((baseline_value - sml_value) / baseline_value) * 100
                    
                    # Add to the row
                    row += f" {sml_value:.3f} | {improvement:+.2f}% |"
                
                f.write(row + "\n")
            
            f.write("\n")
    
    print(f"Saved improvement table to {output_path}")

def create_sml_vs_baseline_improvement_plot(experiments, baseline_exp, output_dir):
    """
    Create plots showing the percentage improvement of each SML method over the baseline GA
    
    Args:
        experiments (dict): Dictionary of experiments and their results
        baseline_exp (str): Name of the experiment to use as the baseline
        output_dir (str): Directory to save the plots
    """
    if baseline_exp not in experiments:
        print(f"Error: Baseline experiment '{baseline_exp}' not found")
        return
    
    # Get baseline data
    baseline_percentages = experiments[baseline_exp]['reduction_percentages']
    
    # Metrics to analyze
    metrics = [
        ("RMSE", "rmse_ga", "rmse_sml"),
        ("MAE", "mae_ga", "mae_sml"),
        ("AbsRel", "absrel_ga", "absrel_sml"),
        ("iRMSE", "inv_rmse_ga", "inv_rmse_sml"),
        ("iMAE", "inv_mae_ga", "inv_mae_sml"),
        ("iAbsRel", "inv_absrel_ga", "inv_absrel_sml")
    ]
    
    for metric_name, ga_key, sml_key in metrics:
        plt.figure(figsize=(12, 8))
        
        palette = sns.color_palette("tab10", len(experiments))
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*']
        
        baseline_values = np.array(experiments[baseline_exp][ga_key])
        
        for i, (exp_name, results) in enumerate(experiments.items()):
            percentages = results['reduction_percentages']
            sml_values = np.array(results[sml_key])
            
            # Match percentages with baseline if different
            if len(percentages) != len(baseline_percentages) or not np.array_equal(percentages, baseline_percentages):
                # Interpolate to match baseline percentages
                from scipy.interpolate import interp1d
                f = interp1d(percentages, sml_values, kind='linear', fill_value='extrapolate')
                sml_values = f(baseline_percentages)
                percentages = baseline_percentages
            
            # Calculate percentage improvement
            # For these metrics, lower is better
            improvement = ((baseline_values - sml_values) / baseline_values) * 100
            
            plt.plot(percentages, improvement, 
                    color=palette[i], 
                    marker=markers[i % len(markers)], 
                    markersize=8,
                    linewidth=2,
                    label=exp_name)
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Point Reduction Percentage (%)', fontsize=14)
        plt.ylabel('Improvement Over Baseline GA (%)', fontsize=14)
        plt.title(f'Percentage Improvement in {metric_name} vs. Baseline GA', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        output_path = os.path.join(output_dir, f'baseline_improvement_{metric_name.lower()}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {metric_name} improvement vs baseline plot to {output_path}")

def compare_all_metrics_combined(experiments, baseline_exp, output_path):
    """
    Create a combined plot showing multiple metrics for a single experiment compared to baseline
    
    Args:
        experiments (dict): Dictionary of experiments and their results
        baseline_exp (str): Name of the experiment to use as the baseline
        output_path (str): Path to save the plot
    """
    if baseline_exp not in experiments:
        print(f"Error: Baseline experiment '{baseline_exp}' not found")
        return
    
    # Define metrics to analyze
    metrics = [
        ("RMSE", "rmse_ga", "rmse_sml"),
        ("MAE", "mae_ga", "mae_sml"),
        ("AbsRel", "absrel_ga", "absrel_sml")
    ]
    
    # Create subplots for each experiment
    n_experiments = len(experiments)
    fig, axes = plt.subplots(n_experiments, 3, figsize=(18, 6*n_experiments), sharex=True)
    
    if n_experiments == 1:
        axes = np.array([axes])  # Make it 2D for consistent indexing
    
    # Get baseline data
    baseline_percentages = experiments[baseline_exp]['reduction_percentages']
    
    # For each experiment and metric
    for i, (exp_name, results) in enumerate(experiments.items()):
        percentages = results['reduction_percentages']
        
        for j, (metric_name, ga_key, sml_key) in enumerate(metrics):
            ax = axes[i, j]
            
            # Plot baseline GA
            baseline_values = experiments[baseline_exp][ga_key]
            ax.plot(baseline_percentages, baseline_values, 
                   color='black', 
                   linestyle='--',
                   linewidth=2,
                   marker='o',
                   markersize=6,
                   label=f"Baseline GA")
            
            # Plot this experiment's SML
            ax.plot(percentages, results[sml_key], 
                   color='blue', 
                   linestyle='-',
                   linewidth=2,
                   marker='s',
                   markersize=6,
                   label=f"{exp_name} SML")
            
            ax.set_title(f"{exp_name}: {metric_name}", fontsize=14)
            ax.set_xlabel('Point Reduction Percentage (%)', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved combined metrics plot to {output_path}")

def analyze_experiments(args):
    """
    Main function to analyze experiments against a baseline
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all JSON files
    if args.json_files:
        json_files = args.json_files
    else:
        # Find all JSON files in the specified directory
        json_files = []
        for root, _, files in os.walk(args.json_dir):
            for file in files:
                if file.endswith('_results.json'):
                    json_files.append(os.path.join(root, file))
    
    if not json_files:
        print("No JSON files found. Please specify valid files or directory.")
        return
    
    # Load experiment results
    experiments = load_experiment_results(json_files)
    
    if not experiments:
        print("No valid experiment results found.")
        return
    
    # Check if baseline exists
    if args.baseline_exp not in experiments:
        print(f"Error: Baseline experiment '{args.baseline_exp}' not found in loaded experiments.")
        print(f"Available experiments: {list(experiments.keys())}")
        return
    
    # Plot comparisons for each metric
    metrics = [
        ("RMSE vs. Baseline GA", "rmse_ga", "rmse_sml", "RMSE"),
        ("MAE vs. Baseline GA", "mae_ga", "mae_sml", "MAE"),
        ("AbsRel vs. Baseline GA", "absrel_ga", "absrel_sml", "AbsRel"),
        ("iRMSE vs. Baseline GA", "inv_rmse_ga", "inv_rmse_sml", "iRMSE"),
        ("iMAE vs. Baseline GA", "inv_mae_ga", "inv_mae_sml", "iMAE"),
        ("iAbsRel vs. Baseline GA", "inv_absrel_ga", "inv_absrel_sml", "iAbsRel")
    ]
    
    for title, metric_ga, metric_sml, ylabel in metrics:
        output_path = os.path.join(args.output_dir, f'{ylabel.lower()}_vs_baseline.png')
        compare_against_baseline(experiments, args.baseline_exp, metric_ga, metric_sml, title, ylabel, output_path)
    
    # Create improvement table
    improvement_path = os.path.join(args.output_dir, 'improvement_vs_baseline.md')
    create_improvement_table(experiments, args.baseline_exp, improvement_path)
    
    # Create improvement plots
    create_sml_vs_baseline_improvement_plot(experiments, args.baseline_exp, args.output_dir)
    
    # Create combined metrics plot
    combined_path = os.path.join(args.output_dir, 'combined_metrics.png')
    compare_all_metrics_combined(experiments, args.baseline_exp, combined_path)
    
    print(f"All analysis results have been saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze VI-Depth experiments against a baseline GA")
    
    # Define input methods: either specific JSON files or a directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--json_files', type=str, nargs='+',
                        help="Paths to JSON result files")
    input_group.add_argument('--json_dir', type=str,
                        help="Directory containing JSON result files")
    
    parser.add_argument('--baseline_exp', type=str, required=True,
                        help="Name of the experiment to use as the baseline")
    parser.add_argument('--output_dir', type=str, default="baseline_analysis",
                        help="Directory to save analysis results")
    
    args = parser.parse_args()
    analyze_experiments(args)
    
#     python baseline_comparison.py \
#   --json_dir ./eval_results \
#   --baseline_exp exp1 \
#   --output_dir baseline_analysis