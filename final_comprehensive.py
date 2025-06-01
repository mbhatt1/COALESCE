#!/usr/bin/env python3
"""
Final comprehensive analysis with 20 runs per experiment for robust statistics.
"""

import os
import sys
import json
import logging
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_single_experiment(duration: int, agents: int, contractors: int, run_id: int):
    """Run a single experiment and extract results."""
    
    cmd = [
        sys.executable, "main.py",
        "--duration", str(duration),
        "--agents", str(agents),
        "--contractors", str(contractors),
        "--config", "config/default_config.yaml"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd="COALESCE")
        
        if result.returncode != 0:
            return None
        
        # Parse metrics from output
        output_lines = result.stdout.split('\n')
        metrics = {}
        
        for line in output_lines:
            if "Average Cost Reduction:" in line:
                metrics['cost_reduction'] = float(line.split(':')[1].strip().replace('%', ''))
            elif "Average Time Savings:" in line:
                metrics['time_savings'] = float(line.split(':')[1].strip().replace('%', ''))
            elif "Total Tasks Processed:" in line:
                metrics['total_tasks'] = int(line.split(':')[1].strip())
        
        # Read executive summary for additional metrics
        try:
            with open("COALESCE/output/reports/executive_summary.md", 'r') as f:
                summary = f.read()
                
            for line in summary.split('\n'):
                if "**Outsourcing Rate:**" in line:
                    rate_str = line.split('**Outsourcing Rate:**')[1].strip()
                    metrics['outsourcing_rate'] = float(rate_str.replace('%', ''))
                elif "**Average TOPSIS Score:**" in line:
                    score_str = line.split('**Average TOPSIS Score:**')[1].strip()
                    metrics['topsis_score'] = float(score_str)
        except:
            metrics['outsourcing_rate'] = 0.0
            metrics['topsis_score'] = 0.0
        
        return {
            'duration': duration,
            'agents': agents,
            'contractors': contractors,
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
    except:
        return None

def main():
    """Run comprehensive analysis with 20 runs per experiment."""
    
    logger.info("Starting comprehensive COALESCE analysis with 20 runs per experiment...")
    
    # Full experiment configurations
    duration_configs = [
        (1, 15, 30), (3, 15, 30), (5, 15, 30), (7, 15, 30), (10, 15, 30),
        (15, 15, 30), (20, 15, 30), (25, 15, 30), (30, 15, 30)
    ]
    
    agent_configs = [
        (7, 5, 10), (7, 10, 20), (7, 15, 30), (7, 20, 40), 
        (7, 25, 50), (7, 30, 60), (7, 40, 80), (7, 50, 100)
    ]
    
    all_results = []
    total_experiments = len(duration_configs) + len(agent_configs)
    experiment_count = 0
    
    # Duration experiments
    logger.info("Running duration scaling experiments...")
    for i, (duration, agents, contractors) in enumerate(duration_configs, 1):
        experiment_count += 1
        exp_id = f'dur_{i:02d}'
        logger.info(f"Experiment {experiment_count}/{total_experiments}: {exp_id} - {duration}d, {agents}a, {contractors}c")
        
        for run in range(20):
            if run % 5 == 0:
                logger.info(f"  Run {run + 1}/20")
            
            result = run_single_experiment(duration, agents, contractors, run + 1)
            if result:
                result['experiment_type'] = 'duration'
                result['experiment_id'] = exp_id
                all_results.append(result)
    
    # Agent scale experiments
    logger.info("Running agent scaling experiments...")
    for i, (duration, agents, contractors) in enumerate(agent_configs, 1):
        experiment_count += 1
        exp_id = f'agt_{i:02d}'
        logger.info(f"Experiment {experiment_count}/{total_experiments}: {exp_id} - {duration}d, {agents}a, {contractors}c")
        
        for run in range(20):
            if run % 5 == 0:
                logger.info(f"  Run {run + 1}/20")
            
            result = run_single_experiment(duration, agents, contractors, run + 1)
            if result:
                result['experiment_type'] = 'agent_scale'
                result['experiment_id'] = exp_id
                all_results.append(result)
    
    # Analyze results
    if not all_results:
        logger.error("No successful runs!")
        return
    
    df = pd.DataFrame(all_results)
    
    # Calculate aggregate statistics
    aggregate_stats = {}
    for metric in ['cost_reduction', 'time_savings', 'outsourcing_rate', 'topsis_score']:
        if metric in df.columns:
            aggregate_stats[metric] = {
                'mean': df[metric].mean(),
                'std': df[metric].std(),
                'min': df[metric].min(),
                'max': df[metric].max(),
                'median': df[metric].median()
            }
    
    # Calculate per-experiment statistics
    experiment_stats = df.groupby('experiment_id').agg({
        'cost_reduction': ['mean', 'std', 'min', 'max'],
        'time_savings': ['mean', 'std', 'min', 'max'],
        'outsourcing_rate': ['mean', 'std', 'min', 'max'],
        'topsis_score': ['mean', 'std', 'min', 'max'],
        'total_tasks': 'mean',
        'duration': 'first',
        'agents': 'first'
    }).round(1)
    
    # Save results
    output_dir = Path("data/comprehensive_analysis_20")
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / "comprehensive_results.csv", index=False)
    experiment_stats.to_csv(output_dir / "experiment_statistics.csv")
    
    with open(output_dir / "aggregate_statistics.json", 'w') as f:
        json.dump(aggregate_stats, f, indent=2)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(experiment_stats)
    with open(output_dir / "results_table.tex", 'w') as f:
        f.write(latex_table)
    
    # Print summary
    logger.info("Comprehensive analysis complete!")
    logger.info(f"Total successful runs: {len(df)}")
    logger.info(f"Results saved to: {output_dir}")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE COALESCE ANALYSIS RESULTS (20 runs per experiment)")
    print("="*80)
    print(f"Total Runs: {len(df)}")
    
    for metric, stats in aggregate_stats.items():
        print(f"{metric.replace('_', ' ').title()}: {stats['mean']:.1f} ± {stats['std']:.1f} (range: {stats['min']:.1f}-{stats['max']:.1f})")
    
    print("="*80)
    
    return aggregate_stats, experiment_stats, df

def generate_latex_table(experiment_stats):
    """Generate LaTeX table for the paper."""
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{COALESCE Performance Results - Comprehensive Validation (20 runs per experiment)}
\\label{tab:comprehensive_results}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{|l|c|c|c|c|c|c|}
\\hline
\\textbf{Experiment} & \\textbf{Duration} & \\textbf{Agents} & \\textbf{Cost Reduction} & \\textbf{Time Savings} & \\textbf{Outsourcing Rate} & \\textbf{TOPSIS Score} \\\\
\\textbf{ID} & \\textbf{(days)} & \\textbf{(count)} & \\textbf{(\\% ± σ)} & \\textbf{(\\% ± σ)} & \\textbf{(\\% ± σ)} & \\textbf{(± σ)} \\\\
\\hline
\\multicolumn{7}{|c|}{\\textbf{Duration Scaling Experiments}} \\\\
\\hline
"""
    
    # Duration experiments
    duration_experiments = [f'dur_{i:02d}' for i in range(1, 10)]
    
    for exp_id in duration_experiments:
        if exp_id in experiment_stats.index:
            stats = experiment_stats.loc[exp_id]
            duration = int(stats[('duration', 'first')])
            agents = int(stats[('agents', 'first')])
            
            cost_mean = stats[('cost_reduction', 'mean')]
            cost_std = stats[('cost_reduction', 'std')]
            time_mean = stats[('time_savings', 'mean')]
            time_std = stats[('time_savings', 'std')]
            out_mean = stats[('outsourcing_rate', 'mean')]
            out_std = stats[('outsourcing_rate', 'std')]
            topsis_mean = stats[('topsis_score', 'mean')]
            topsis_std = stats[('topsis_score', 'std')]
            
            latex_table += f"{exp_id} & {duration} & {agents} & {cost_mean:.1f} ± {cost_std:.1f} & {time_mean:.1f} ± {time_std:.1f} & {out_mean:.1f} ± {out_std:.1f} & {topsis_mean:.3f} ± {topsis_std:.3f} \\\\\n"
    
    latex_table += """\\hline
\\multicolumn{7}{|c|}{\\textbf{Agent Scale Experiments}} \\\\
\\hline
"""
    
    # Agent experiments
    agent_experiments = [f'agt_{i:02d}' for i in range(1, 9)]
    
    for exp_id in agent_experiments:
        if exp_id in experiment_stats.index:
            stats = experiment_stats.loc[exp_id]
            duration = int(stats[('duration', 'first')])
            agents = int(stats[('agents', 'first')])
            
            cost_mean = stats[('cost_reduction', 'mean')]
            cost_std = stats[('cost_reduction', 'std')]
            time_mean = stats[('time_savings', 'mean')]
            time_std = stats[('time_savings', 'std')]
            out_mean = stats[('outsourcing_rate', 'mean')]
            out_std = stats[('outsourcing_rate', 'std')]
            topsis_mean = stats[('topsis_score', 'mean')]
            topsis_std = stats[('topsis_score', 'std')]
            
            latex_table += f"{exp_id} & {duration} & {agents} & {cost_mean:.1f} ± {cost_std:.1f} & {time_mean:.1f} ± {time_std:.1f} & {out_mean:.1f} ± {out_std:.1f} & {topsis_mean:.3f} ± {topsis_std:.3f} \\\\\n"
    
    latex_table += """\\hline
\\end{tabular}%
}
\\end{table}
"""
    
    return latex_table

if __name__ == "__main__":
    main()