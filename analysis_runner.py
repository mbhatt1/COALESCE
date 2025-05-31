#!/usr/bin/env python3
"""
Analysis runner for COALESCE framework validation.

This script runs systematic experiments to analyze:
1. Duration vs Cost Reduction
2. Number of Agents vs Cost Reduction

Generates plots and statistical analysis for paper validation.
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

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AnalysisRunner:
    """Run systematic analysis experiments for COALESCE validation."""
    
    def __init__(self, output_dir="analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "analysis.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_single_experiment(self, duration: int, agents: int, contractors: int, 
                            experiment_id: str) -> Dict:
        """Run a single simulation experiment."""
        
        self.logger.info(f"Running experiment {experiment_id}: {duration}d, {agents}a, {contractors}c")
        
        # Create output directory for this experiment
        exp_output_dir = self.output_dir / f"exp_{experiment_id}"
        exp_output_dir.mkdir(exist_ok=True)
        
        # Build command
        cmd = [
            "python3", "main.py",
            "--duration", str(duration),
            "--agents", str(agents),
            "--contractors", str(contractors),
            "--output-dir", str(exp_output_dir)
        ]
        
        try:
            # Run simulation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=Path(__file__).parent
            )
            
            if result.returncode == 0:
                # Parse results
                performance_file = exp_output_dir / "reports" / "performance_benchmarks.json"
                if performance_file.exists():
                    with open(performance_file, 'r') as f:
                        performance_data = json.load(f)
                    
                    # Extract key metrics
                    experiment_result = {
                        'experiment_id': experiment_id,
                        'duration': duration,
                        'agents': agents,
                        'contractors': contractors,
                        'success': True,
                        'cost_reduction_pct': performance_data['cost_performance']['average_cost_reduction_percent'],
                        'time_savings_pct': performance_data['time_performance']['average_time_savings_percent'],
                        'throughput': performance_data['time_performance']['throughput_tasks_per_hour'],
                        'outsourcing_rate': performance_data['decision_quality']['outsourcing_rate'],
                        'topsis_score': performance_data['decision_quality']['average_topsis_score'],
                        'confidence': performance_data['decision_quality']['average_confidence'],
                        'total_tasks': performance_data['simulation_metadata']['total_tasks'],
                        'output_dir': str(exp_output_dir)
                    }
                    
                else:
                    experiment_result = {
                        'experiment_id': experiment_id,
                        'success': False,
                        'error': 'No performance data generated'
                    }
            else:
                experiment_result = {
                    'experiment_id': experiment_id,
                    'success': False,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            experiment_result = {
                'experiment_id': experiment_id,
                'success': False,
                'error': 'Simulation timeout'
            }
        except Exception as e:
            experiment_result = {
                'experiment_id': experiment_id,
                'success': False,
                'error': str(e)
            }
        
        self.results.append(experiment_result)
        return experiment_result
    
    def run_duration_analysis(self):
        """Run experiments varying simulation duration."""
        self.logger.info("Starting Duration vs Cost Reduction analysis...")
        
        # Fixed parameters
        base_agents = 15
        base_contractors = 30
        
        # Varying durations
        durations = [1, 3, 5, 7, 10, 15, 20, 25, 30]
        
        duration_results = []
        for i, duration in enumerate(durations):
            result = self.run_single_experiment(
                duration=duration,
                agents=base_agents,
                contractors=base_contractors,
                experiment_id=f"dur_{i+1:02d}"
            )
            if result['success']:
                duration_results.append(result)
        
        return duration_results
    
    def run_agent_analysis(self):
        """Run experiments varying number of agents."""
        self.logger.info("Starting Agent Count vs Cost Reduction analysis...")
        
        # Fixed parameters
        base_duration = 7
        
        # Varying agent counts (keeping 2:1 contractor ratio)
        agent_configs = [
            (5, 10),    # 5 agents, 10 contractors
            (10, 20),   # 10 agents, 20 contractors
            (15, 30),   # 15 agents, 30 contractors
            (20, 40),   # 20 agents, 40 contractors
            (25, 50),   # 25 agents, 50 contractors
            (30, 60),   # 30 agents, 60 contractors
            (40, 80),   # 40 agents, 80 contractors
            (50, 100),  # 50 agents, 100 contractors
        ]
        
        agent_results = []
        for i, (agents, contractors) in enumerate(agent_configs):
            result = self.run_single_experiment(
                duration=base_duration,
                agents=agents,
                contractors=contractors,
                experiment_id=f"agt_{i+1:02d}"
            )
            if result['success']:
                agent_results.append(result)
        
        return agent_results
    
    def create_duration_plot(self, duration_results: List[Dict]):
        """Create duration vs cost reduction plot."""
        if not duration_results:
            self.logger.warning("No duration results to plot")
            return
        
        df = pd.DataFrame(duration_results)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Duration vs Cost Reduction
        ax1.scatter(df['duration'], df['cost_reduction_pct'], s=100, alpha=0.7, color='blue')
        ax1.plot(df['duration'], df['cost_reduction_pct'], '--', alpha=0.5, color='blue')
        ax1.set_xlabel('Simulation Duration (days)')
        ax1.set_ylabel('Cost Reduction (%)')
        ax1.set_title('Duration vs Cost Reduction')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['duration'], df['cost_reduction_pct'], 1)
        p = np.poly1d(z)
        ax1.plot(df['duration'], p(df['duration']), "r--", alpha=0.8, 
                label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        ax1.legend()
        
        # Duration vs Time Savings
        ax2.scatter(df['duration'], df['time_savings_pct'], s=100, alpha=0.7, color='green')
        ax2.plot(df['duration'], df['time_savings_pct'], '--', alpha=0.5, color='green')
        ax2.set_xlabel('Simulation Duration (days)')
        ax2.set_ylabel('Time Savings (%)')
        ax2.set_title('Duration vs Time Savings')
        ax2.grid(True, alpha=0.3)
        
        # Duration vs Throughput
        ax3.scatter(df['duration'], df['throughput'], s=100, alpha=0.7, color='orange')
        ax3.plot(df['duration'], df['throughput'], '--', alpha=0.5, color='orange')
        ax3.set_xlabel('Simulation Duration (days)')
        ax3.set_ylabel('Throughput (tasks/hour)')
        ax3.set_title('Duration vs System Throughput')
        ax3.grid(True, alpha=0.3)
        
        # Duration vs Outsourcing Rate
        ax4.scatter(df['duration'], df['outsourcing_rate'] * 100, s=100, alpha=0.7, color='purple')
        ax4.plot(df['duration'], df['outsourcing_rate'] * 100, '--', alpha=0.5, color='purple')
        ax4.set_xlabel('Simulation Duration (days)')
        ax4.set_ylabel('Outsourcing Rate (%)')
        ax4.set_title('Duration vs Outsourcing Rate')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('COALESCE Performance vs Simulation Duration', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "duration_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Duration analysis plot saved to {plot_path}")
        return plot_path
    
    def create_agent_plot(self, agent_results: List[Dict]):
        """Create agent count vs cost reduction plot."""
        if not agent_results:
            self.logger.warning("No agent results to plot")
            return
        
        df = pd.DataFrame(agent_results)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Agent Count vs Cost Reduction
        ax1.scatter(df['agents'], df['cost_reduction_pct'], s=100, alpha=0.7, color='blue')
        ax1.plot(df['agents'], df['cost_reduction_pct'], '--', alpha=0.5, color='blue')
        ax1.set_xlabel('Number of Client Agents')
        ax1.set_ylabel('Cost Reduction (%)')
        ax1.set_title('Agent Count vs Cost Reduction')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['agents'], df['cost_reduction_pct'], 1)
        p = np.poly1d(z)
        ax1.plot(df['agents'], p(df['agents']), "r--", alpha=0.8, 
                label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        ax1.legend()
        
        # Agent Count vs Time Savings
        ax2.scatter(df['agents'], df['time_savings_pct'], s=100, alpha=0.7, color='green')
        ax2.plot(df['agents'], df['time_savings_pct'], '--', alpha=0.5, color='green')
        ax2.set_xlabel('Number of Client Agents')
        ax2.set_ylabel('Time Savings (%)')
        ax2.set_title('Agent Count vs Time Savings')
        ax2.grid(True, alpha=0.3)
        
        # Agent Count vs Total Tasks
        ax3.scatter(df['agents'], df['total_tasks'], s=100, alpha=0.7, color='orange')
        ax3.plot(df['agents'], df['total_tasks'], '--', alpha=0.5, color='orange')
        ax3.set_xlabel('Number of Client Agents')
        ax3.set_ylabel('Total Tasks Processed')
        ax3.set_title('Agent Count vs Task Volume')
        ax3.grid(True, alpha=0.3)
        
        # Agent Count vs Outsourcing Rate
        ax4.scatter(df['agents'], df['outsourcing_rate'] * 100, s=100, alpha=0.7, color='purple')
        ax4.plot(df['agents'], df['outsourcing_rate'] * 100, '--', alpha=0.5, color='purple')
        ax4.set_xlabel('Number of Client Agents')
        ax4.set_ylabel('Outsourcing Rate (%)')
        ax4.set_title('Agent Count vs Outsourcing Rate')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('COALESCE Performance vs Agent Count', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "agent_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Agent analysis plot saved to {plot_path}")
        return plot_path
    
    def create_combined_analysis_plot(self, duration_results: List[Dict], agent_results: List[Dict]):
        """Create combined analysis plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Duration analysis
        if duration_results:
            df_dur = pd.DataFrame(duration_results)
            ax1.scatter(df_dur['duration'], df_dur['cost_reduction_pct'], 
                       s=120, alpha=0.7, color='blue', label='Actual Data')
            
            # Fit polynomial trend
            z = np.polyfit(df_dur['duration'], df_dur['cost_reduction_pct'], 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(df_dur['duration'].min(), df_dur['duration'].max(), 100)
            ax1.plot(x_smooth, p(x_smooth), "r-", alpha=0.8, linewidth=2,
                    label=f'Polynomial Fit')
            
            ax1.set_xlabel('Simulation Duration (days)', fontsize=12)
            ax1.set_ylabel('Cost Reduction (%)', fontsize=12)
            ax1.set_title('Duration vs Cost Reduction', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Agent analysis
        if agent_results:
            df_agt = pd.DataFrame(agent_results)
            ax2.scatter(df_agt['agents'], df_agt['cost_reduction_pct'], 
                       s=120, alpha=0.7, color='green', label='Actual Data')
            
            # Fit polynomial trend
            z = np.polyfit(df_agt['agents'], df_agt['cost_reduction_pct'], 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(df_agt['agents'].min(), df_agt['agents'].max(), 100)
            ax2.plot(x_smooth, p(x_smooth), "r-", alpha=0.8, linewidth=2,
                    label=f'Polynomial Fit')
            
            ax2.set_xlabel('Number of Client Agents', fontsize=12)
            ax2.set_ylabel('Cost Reduction (%)', fontsize=12)
            ax2.set_title('Agent Count vs Cost Reduction', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.suptitle('COALESCE Framework Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "combined_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_analysis_report(self, duration_results: List[Dict], agent_results: List[Dict]):
        """Generate comprehensive analysis report."""
        report_path = self.output_dir / "analysis_report.md"
        
        # Calculate statistics
        dur_stats = self._calculate_statistics(duration_results, 'duration')
        agt_stats = self._calculate_statistics(agent_results, 'agents')
        
        content = f"""# COALESCE Framework Performance Analysis

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This analysis examines the relationship between simulation parameters and COALESCE framework performance through systematic experimentation.

## Duration Analysis

### Experiments Conducted
- **Total Experiments:** {len(duration_results)}
- **Duration Range:** {min(r['duration'] for r in duration_results)} - {max(r['duration'] for r in duration_results)} days
- **Fixed Parameters:** 15 client agents, 30 contractor agents

### Key Findings

#### Cost Reduction vs Duration
{dur_stats}

#### Performance Trends
- **Correlation (Duration vs Cost Reduction):** {self._calculate_correlation(duration_results, 'duration', 'cost_reduction_pct'):.3f}
- **Correlation (Duration vs Time Savings):** {self._calculate_correlation(duration_results, 'duration', 'time_savings_pct'):.3f}

## Agent Count Analysis

### Experiments Conducted
- **Total Experiments:** {len(agent_results)}
- **Agent Range:** {min(r['agents'] for r in agent_results)} - {max(r['agents'] for r in agent_results)} agents
- **Fixed Parameters:** 7-day duration, 2:1 contractor ratio

### Key Findings

#### Cost Reduction vs Agent Count
{agt_stats}

#### Performance Trends
- **Correlation (Agents vs Cost Reduction):** {self._calculate_correlation(agent_results, 'agents', 'cost_reduction_pct'):.3f}
- **Correlation (Agents vs Time Savings):** {self._calculate_correlation(agent_results, 'agents', 'time_savings_pct'):.3f}

## Statistical Analysis

### Duration Results
"""
        
        if duration_results:
            df_dur = pd.DataFrame(duration_results)
            content += f"""
- **Mean Cost Reduction:** {df_dur['cost_reduction_pct'].mean():.1f}%
- **Std Dev Cost Reduction:** {df_dur['cost_reduction_pct'].std():.1f}%
- **Min Cost Reduction:** {df_dur['cost_reduction_pct'].min():.1f}%
- **Max Cost Reduction:** {df_dur['cost_reduction_pct'].max():.1f}%
"""
        
        content += "\n### Agent Count Results\n"
        
        if agent_results:
            df_agt = pd.DataFrame(agent_results)
            content += f"""
- **Mean Cost Reduction:** {df_agt['cost_reduction_pct'].mean():.1f}%
- **Std Dev Cost Reduction:** {df_agt['cost_reduction_pct'].std():.1f}%
- **Min Cost Reduction:** {df_agt['cost_reduction_pct'].min():.1f}%
- **Max Cost Reduction:** {df_agt['cost_reduction_pct'].max():.1f}%
"""
        
        content += f"""
## Conclusions

1. **Duration Impact:** {"Longer simulations show improved cost reduction" if self._calculate_correlation(duration_results, 'duration', 'cost_reduction_pct') > 0.3 else "Duration has limited impact on cost reduction"}

2. **Scale Impact:** {"Larger agent populations improve performance" if self._calculate_correlation(agent_results, 'agents', 'cost_reduction_pct') > 0.3 else "Agent count has limited impact on performance"}

3. **Framework Validation:** The COALESCE framework demonstrates {"consistent" if min(df_dur['cost_reduction_pct'].min() if duration_results else 0, df_agt['cost_reduction_pct'].min() if agent_results else 0) > 10 else "variable"} performance across different configurations.

## Recommendations

1. **Optimal Duration:** {"15+ days recommended for stable results" if duration_results and max(r['cost_reduction_pct'] for r in duration_results) > 25 else "Shorter durations may be sufficient"}

2. **Optimal Scale:** {"25+ agents recommended for best performance" if agent_results and any(r['cost_reduction_pct'] > 25 for r in agent_results if r['agents'] >= 25) else "Smaller deployments may be adequate"}

---
*Generated by COALESCE Analysis Framework*
"""
        
        with open(report_path, 'w') as f:
            f.write(content)
        
        return report_path
    
    def _calculate_statistics(self, results: List[Dict], param_name: str) -> str:
        """Calculate statistics for a parameter."""
        if not results:
            return "No data available"
        
        df = pd.DataFrame(results)
        param_values = df[param_name]
        cost_values = df['cost_reduction_pct']
        
        correlation = np.corrcoef(param_values, cost_values)[0, 1]
        
        return f"""
- **Parameter Range:** {param_values.min()} - {param_values.max()}
- **Cost Reduction Range:** {cost_values.min():.1f}% - {cost_values.max():.1f}%
- **Correlation Coefficient:** {correlation:.3f}
- **Trend:** {"Positive" if correlation > 0.1 else "Negative" if correlation < -0.1 else "Neutral"}
"""
    
    def _calculate_correlation(self, results: List[Dict], x_param: str, y_param: str) -> float:
        """Calculate correlation between two parameters."""
        if not results:
            return 0.0
        
        df = pd.DataFrame(results)
        return np.corrcoef(df[x_param], df[y_param])[0, 1]
    
    def run_full_analysis(self):
        """Run complete analysis suite."""
        self.logger.info("Starting comprehensive COALESCE analysis...")
        
        # Run experiments
        duration_results = self.run_duration_analysis()
        agent_results = self.run_agent_analysis()
        
        # Save raw results
        all_results = {
            'duration_experiments': duration_results,
            'agent_experiments': agent_results,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_experiments': len(duration_results) + len(agent_results)
            }
        }
        
        results_file = self.output_dir / "analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate plots
        duration_plot = self.create_duration_plot(duration_results)
        agent_plot = self.create_agent_plot(agent_results)
        combined_plot = self.create_combined_analysis_plot(duration_results, agent_results)
        
        # Generate report
        report = self.generate_analysis_report(duration_results, agent_results)
        
        # Save summary CSV
        if duration_results:
            df_dur = pd.DataFrame(duration_results)
            df_dur.to_csv(self.output_dir / "duration_results.csv", index=False)
        
        if agent_results:
            df_agt = pd.DataFrame(agent_results)
            df_agt.to_csv(self.output_dir / "agent_results.csv", index=False)
        
        self.logger.info("Analysis complete!")
        print("\n" + "="*80)
        print("COALESCE ANALYSIS COMPLETE")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")
        print(f"Duration experiments: {len(duration_results)}")
        print(f"Agent experiments: {len(agent_results)}")
        print(f"Plots generated: {duration_plot}, {agent_plot}, {combined_plot}")
        print(f"Report: {report}")
        print("="*80)


def main():
    runner = AnalysisRunner("analysis_results")
    runner.run_full_analysis()


if __name__ == "__main__":
    main()