#!/usr/bin/env python3
"""
Comprehensive analysis with 100 runs per experiment configuration.
Uses the existing main.py infrastructure for reliability.
"""

import os
import sys
import json
import logging
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveAnalysis:
    """Run comprehensive analysis with 100 runs per experiment."""
    
    def __init__(self):
        self.output_dir = Path("data/comprehensive_analysis_100")
        self.output_dir.mkdir(exist_ok=True)
        
        # Experiment configurations
        self.duration_configs = [
            (1, 15, 30), (3, 15, 30), (5, 15, 30), (7, 15, 30), (10, 15, 30),
            (15, 15, 30), (20, 15, 30), (25, 15, 30), (30, 15, 30)
        ]
        
        self.agent_configs = [
            (7, 5, 10), (7, 10, 20), (7, 15, 30), (7, 20, 40), 
            (7, 25, 50), (7, 30, 60), (7, 40, 80), (7, 50, 100)
        ]
        
        self.all_results = []
    
    def run_single_experiment(self, duration: int, agents: int, contractors: int, run_id: int) -> Dict:
        """Run a single experiment and extract results."""
        
        # Run the simulation
        cmd = [
            sys.executable, "main.py",
            "--duration", str(duration),
            "--agents", str(agents),
            "--contractors", str(contractors),
            "--config", "config/content_creation_config.yaml"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"Simulation failed: {result.stderr}")
                return None
            
            # Parse the output to extract metrics
            output_lines = result.stdout.split('\n')
            metrics = {}
            
            for line in output_lines:
                if "Average Cost Reduction:" in line:
                    metrics['cost_reduction'] = float(line.split(':')[1].strip().replace('%', ''))
                elif "Average Time Savings:" in line:
                    metrics['time_savings'] = float(line.split(':')[1].strip().replace('%', ''))
                elif "System Throughput:" in line:
                    metrics['throughput'] = float(line.split(':')[1].strip().split()[0])
                elif "Total Tasks Processed:" in line:
                    metrics['total_tasks'] = int(line.split(':')[1].strip())
            
            # Read the executive summary for more detailed metrics
            try:
                with open("output/reports/executive_summary.md", 'r') as f:
                    summary = f.read()
                    
                for line in summary.split('\n'):
                    if "**Outsourcing Rate:**" in line:
                        rate_str = line.split('**Outsourcing Rate:**')[1].strip()
                        metrics['outsourcing_rate'] = float(rate_str.replace('%', ''))
                    elif "**Average TOPSIS Score:**" in line:
                        score_str = line.split('**Average TOPSIS Score:**')[1].strip()
                        metrics['topsis_score'] = float(score_str)
            except:
                # Fallback values if file reading fails
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
            
        except subprocess.TimeoutExpired:
            logger.error(f"Simulation timed out for {duration}d, {agents}a, {contractors}c")
            return None
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            return None
    
    def run_duration_experiments(self):
        """Run duration scaling experiments."""
        
        logger.info("Starting duration scaling experiments (100 runs each)...")
        
        for i, (duration, agents, contractors) in enumerate(self.duration_configs, 1):
            exp_id = f'dur_{i:02d}'
            logger.info(f"Experiment {exp_id}: {duration}d, {agents}a, {contractors}c")
            
            for run in range(100):
                if run % 20 == 0:
                    logger.info(f"  Run {run + 1}/100")
                
                result = self.run_single_experiment(duration, agents, contractors, run + 1)
                if result:
                    result['experiment_type'] = 'duration'
                    result['experiment_id'] = exp_id
                    self.all_results.append(result)
    
    def run_agent_experiments(self):
        """Run agent scaling experiments."""
        
        logger.info("Starting agent scaling experiments (100 runs each)...")
        
        for i, (duration, agents, contractors) in enumerate(self.agent_configs, 1):
            exp_id = f'agt_{i:02d}'
            logger.info(f"Experiment {exp_id}: {duration}d, {agents}a, {contractors}c")
            
            for run in range(100):
                if run % 20 == 0:
                    logger.info(f"  Run {run + 1}/100")
                
                result = self.run_single_experiment(duration, agents, contractors, run + 1)
                if result:
                    result['experiment_type'] = 'agent_scale'
                    result['experiment_id'] = exp_id
                    self.all_results.append(result)
    
    def analyze_results(self):
        """Analyze the comprehensive results."""
        
        if not self.all_results:
            logger.error("No results to analyze!")
            return
        
        df = pd.DataFrame(self.all_results)
        
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
            'cost_reduction': ['mean', 'std', 'min', 'max'] if 'cost_reduction' in df.columns else ['count'],
            'time_savings': ['mean', 'std', 'min', 'max'] if 'time_savings' in df.columns else ['count'],
            'outsourcing_rate': ['mean', 'std', 'min', 'max'] if 'outsourcing_rate' in df.columns else ['count'],
            'topsis_score': ['mean', 'std', 'min', 'max'] if 'topsis_score' in df.columns else ['count'],
            'total_tasks': 'mean' if 'total_tasks' in df.columns else 'count',
            'duration': 'first',
            'agents': 'first'
        }).round(1)
        
        return aggregate_stats, experiment_stats, df
    
    def generate_latex_table(self, experiment_stats):
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
                
                # Handle missing columns gracefully
                try:
                    cost_mean = stats[('cost_reduction', 'mean')]
                    cost_std = stats[('cost_reduction', 'std')]
                    time_mean = stats[('time_savings', 'mean')]
                    time_std = stats[('time_savings', 'std')]
                    out_mean = stats[('outsourcing_rate', 'mean')]
                    out_std = stats[('outsourcing_rate', 'std')]
                    topsis_mean = stats[('topsis_score', 'mean')]
                    topsis_std = stats[('topsis_score', 'std')]
                    
                    latex_table += f"{exp_id} & {duration} & {agents} & {cost_mean:.1f} ± {cost_std:.1f} & {time_mean:.1f} ± {time_std:.1f} & {out_mean:.1f} ± {out_std:.1f} & {topsis_mean:.3f} ± {topsis_std:.3f} \\\\\n"
                except:
                    latex_table += f"{exp_id} & {duration} & {agents} & N/A & N/A & N/A & N/A \\\\\n"
        
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
                
                try:
                    cost_mean = stats[('cost_reduction', 'mean')]
                    cost_std = stats[('cost_reduction', 'std')]
                    time_mean = stats[('time_savings', 'mean')]
                    time_std = stats[('time_savings', 'std')]
                    out_mean = stats[('outsourcing_rate', 'mean')]
                    out_std = stats[('outsourcing_rate', 'std')]
                    topsis_mean = stats[('topsis_score', 'mean')]
                    topsis_std = stats[('topsis_score', 'std')]
                    
                    latex_table += f"{exp_id} & {duration} & {agents} & {cost_mean:.1f} ± {cost_std:.1f} & {time_mean:.1f} ± {time_std:.1f} & {out_mean:.1f} ± {out_std:.1f} & {topsis_mean:.3f} ± {topsis_std:.3f} \\\\\n"
                except:
                    latex_table += f"{exp_id} & {duration} & {agents} & N/A & N/A & N/A & N/A \\\\\n"
        
        latex_table += """\\hline
\\end{tabular}%
}
\\end{table}
"""
        
        return latex_table
    
    def save_results(self, aggregate_stats, experiment_stats, df):
        """Save all results and analysis."""
        
        # Save raw data
        df.to_csv(self.output_dir / "comprehensive_results.csv", index=False)
        
        # Save statistics
        with open(self.output_dir / "aggregate_statistics.json", 'w') as f:
            json.dump(aggregate_stats, f, indent=2)
        
        experiment_stats.to_csv(self.output_dir / "experiment_statistics.csv")
        
        # Generate and save LaTeX table
        latex_table = self.generate_latex_table(experiment_stats)
        with open(self.output_dir / "results_table.tex", 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def run_comprehensive_analysis(self):
        """Run the complete comprehensive analysis."""
        
        logger.info("Starting comprehensive COALESCE analysis...")
        logger.info("This will run 1700 total experiments (100 runs × 17 configurations)")
        logger.info("Estimated time: 45-60 minutes")
        
        # Run experiments
        self.run_duration_experiments()
        self.run_agent_experiments()
        
        # Analyze results
        aggregate_stats, experiment_stats, df = self.analyze_results()
        
        # Save results
        self.save_results(aggregate_stats, experiment_stats, df)
        
        # Print summary
        logger.info("Comprehensive analysis complete!")
        logger.info(f"Total successful runs: {len(df)}")
        
        if aggregate_stats:
            print("\n" + "="*80)
            print("COMPREHENSIVE COALESCE ANALYSIS RESULTS")
            print("="*80)
            print(f"Total Runs: {len(df)}")
            
            for metric, stats in aggregate_stats.items():
                print(f"{metric.replace('_', ' ').title()}: {stats['mean']:.1f} ± {stats['std']:.1f}")
            
            print("="*80)
        
        return aggregate_stats, experiment_stats, df

def main():
    """Main execution function."""
    analysis = ComprehensiveAnalysis()
    return analysis.run_comprehensive_analysis()

if __name__ == "__main__":
    main()