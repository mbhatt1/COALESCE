#!/usr/bin/env python3
"""
Extended analysis with 100 runs per experiment for robust statistical validation.
This will generate the definitive results table for the paper.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from src.simulation.simulation_engine import SimulationEngine
from src.config.simulation_config import SimulationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_extended_experiments():
    """Run 100 experiments for each configuration."""
    
    # Experiment configurations
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
    logger.info("Starting extended duration experiments (100 runs each)...")
    for i, (duration, agents, contractors) in enumerate(duration_configs, 1):
        experiment_count += 1
        logger.info(f"Experiment {experiment_count}/{total_experiments}: Duration {duration}d, {agents}a, {contractors}c")
        
        for run in range(100):
            if run % 20 == 0:
                logger.info(f"  Run {run + 1}/100")
                
            config = SimulationConfig(
                simulation_duration_days=duration,
                num_client_agents=agents,
                num_contractor_agents=contractors,
                random_seed=None  # Different seed each run
            )
            
            engine = SimulationEngine(config)
            results = engine.run()
            
            result_data = {
                'experiment_type': 'duration',
                'experiment_id': f'dur_{i:02d}',
                'run_number': run + 1,
                'duration': duration,
                'agents': agents,
                'contractors': contractors,
                'cost_reduction': results.metrics.cost_reduction_percentage,
                'time_savings': results.metrics.time_savings_percentage,
                'outsourcing_rate': results.metrics.outsourcing_rate_percentage,
                'topsis_score': results.metrics.average_topsis_score,
                'total_tasks': results.metrics.total_tasks,
                'timestamp': datetime.now().isoformat()
            }
            all_results.append(result_data)
    
    # Agent scale experiments  
    logger.info("Starting extended agent scale experiments (100 runs each)...")
    for i, (duration, agents, contractors) in enumerate(agent_configs, 1):
        experiment_count += 1
        logger.info(f"Experiment {experiment_count}/{total_experiments}: {duration}d, {agents}a, {contractors}c")
        
        for run in range(100):
            if run % 20 == 0:
                logger.info(f"  Run {run + 1}/100")
                
            config = SimulationConfig(
                simulation_duration_days=duration,
                num_client_agents=agents,
                num_contractor_agents=contractors,
                random_seed=None  # Different seed each run
            )
            
            engine = SimulationEngine(config)
            results = engine.run()
            
            result_data = {
                'experiment_type': 'agent_scale',
                'experiment_id': f'agt_{i:02d}',
                'run_number': run + 1,
                'duration': duration,
                'agents': agents,
                'contractors': contractors,
                'cost_reduction': results.metrics.cost_reduction_percentage,
                'time_savings': results.metrics.time_savings_percentage,
                'outsourcing_rate': results.metrics.outsourcing_rate_percentage,
                'topsis_score': results.metrics.average_topsis_score,
                'total_tasks': results.metrics.total_tasks,
                'timestamp': datetime.now().isoformat()
            }
            all_results.append(result_data)
    
    return all_results

def analyze_extended_results(results):
    """Analyze the extended experimental results."""
    
    df = pd.DataFrame(results)
    
    # Calculate aggregate statistics
    aggregate_stats = {
        'cost_reduction': {
            'mean': df['cost_reduction'].mean(),
            'std': df['cost_reduction'].std(),
            'min': df['cost_reduction'].min(),
            'max': df['cost_reduction'].max(),
            'median': df['cost_reduction'].median()
        },
        'time_savings': {
            'mean': df['time_savings'].mean(),
            'std': df['time_savings'].std(),
            'min': df['time_savings'].min(),
            'max': df['time_savings'].max(),
            'median': df['time_savings'].median()
        },
        'outsourcing_rate': {
            'mean': df['outsourcing_rate'].mean(),
            'std': df['outsourcing_rate'].std(),
            'min': df['outsourcing_rate'].min(),
            'max': df['outsourcing_rate'].max(),
            'median': df['outsourcing_rate'].median()
        },
        'topsis_score': {
            'mean': df['topsis_score'].mean(),
            'std': df['topsis_score'].std(),
            'min': df['topsis_score'].min(),
            'max': df['topsis_score'].max(),
            'median': df['topsis_score'].median()
        }
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
    
    return aggregate_stats, experiment_stats, df

def generate_paper_table(experiment_stats):
    """Generate LaTeX table for the paper."""
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{COALESCE Performance Results - Comprehensive Validation (100 runs per experiment)}
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

def generate_summary_statistics(aggregate_stats):
    """Generate summary statistics for the paper."""
    
    summary = f"""
\\section{{Experimental Results Summary}}

Based on comprehensive validation with 100 runs per experiment configuration, the COALESCE framework demonstrates the following aggregate performance:

\\begin{{itemize}}
\\item \\textbf{{Average Cost Reduction:}} {aggregate_stats['cost_reduction']['mean']:.1f}\\% ± {aggregate_stats['cost_reduction']['std']:.1f}\\% (range: {aggregate_stats['cost_reduction']['min']:.1f}\\% to {aggregate_stats['cost_reduction']['max']:.1f}\\%)
\\item \\textbf{{Average Time Savings:}} {aggregate_stats['time_savings']['mean']:.1f}\\% ± {aggregate_stats['time_savings']['std']:.1f}\\% (range: {aggregate_stats['time_savings']['min']:.1f}\\% to {aggregate_stats['time_savings']['max']:.1f}\\%)
\\item \\textbf{{Average Outsourcing Rate:}} {aggregate_stats['outsourcing_rate']['mean']:.1f}\\% ± {aggregate_stats['outsourcing_rate']['std']:.1f}\\% (range: {aggregate_stats['outsourcing_rate']['min']:.1f}\\% to {aggregate_stats['outsourcing_rate']['max']:.1f}\\%)
\\item \\textbf{{Average TOPSIS Score:}} {aggregate_stats['topsis_score']['mean']:.3f} ± {aggregate_stats['topsis_score']['std']:.3f} (range: {aggregate_stats['topsis_score']['min']:.3f} to {aggregate_stats['topsis_score']['max']:.3f})
\\end{{itemize}}

These results demonstrate the robustness and effectiveness of the COALESCE framework across diverse operational scenarios.
"""
    
    return summary

def main():
    """Main execution function."""
    
    logger.info("Starting comprehensive COALESCE analysis with 100 runs per experiment...")
    logger.info("This will take approximately 30-45 minutes to complete.")
    
    # Run experiments
    results = run_extended_experiments()
    
    # Analyze results
    aggregate_stats, experiment_stats, df = analyze_extended_results(results)
    
    # Save raw results
    output_dir = Path("data/extended_analysis_100")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    df.to_csv(output_dir / "detailed_results.csv", index=False)
    
    with open(output_dir / "aggregate_statistics.json", 'w') as f:
        json.dump(aggregate_stats, f, indent=2)
    
    experiment_stats.to_csv(output_dir / "experiment_statistics.csv")
    
    # Generate LaTeX table
    latex_table = generate_paper_table(experiment_stats)
    with open(output_dir / "results_table.tex", 'w') as f:
        f.write(latex_table)
    
    # Generate summary statistics
    summary = generate_summary_statistics(aggregate_stats)
    with open(output_dir / "summary_statistics.tex", 'w') as f:
        f.write(summary)
    
    # Print summary
    logger.info("Analysis complete!")
    logger.info(f"Total experiments: {len(df)}")
    logger.info(f"Results saved to: {output_dir}")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE COALESCE ANALYSIS RESULTS")
    print("="*80)
    print(f"Total Runs: {len(df)}")
    print(f"Average Cost Reduction: {aggregate_stats['cost_reduction']['mean']:.1f}% ± {aggregate_stats['cost_reduction']['std']:.1f}%")
    print(f"Average Time Savings: {aggregate_stats['time_savings']['mean']:.1f}% ± {aggregate_stats['time_savings']['std']:.1f}%")
    print(f"Average Outsourcing Rate: {aggregate_stats['outsourcing_rate']['mean']:.1f}% ± {aggregate_stats['outsourcing_rate']['std']:.1f}%")
    print(f"Average TOPSIS Score: {aggregate_stats['topsis_score']['mean']:.3f} ± {aggregate_stats['topsis_score']['std']:.3f}")
    print("="*80)

if __name__ == "__main__":
    main()