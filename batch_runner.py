#!/usr/bin/env python3
"""
Batch runner for COALESCE simulations to validate paper claims.

This script runs multiple simulations with different configurations and random seeds
to find both successful and poor-performing scenarios for scientific validation.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.simulation_config import SimulationConfig


class BatchRunner:
    """Run multiple COALESCE simulations and analyze results."""
    
    def __init__(self, output_base_dir="batch_results"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        self.results = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_base_dir / "batch_run.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_single_simulation(self, run_id, config_file, duration=7, agents=15, contractors=30, seed=None):
        """Run a single simulation and capture results."""
        
        self.logger.info(f"Starting simulation run {run_id}")
        
        # Create output directory for this run
        run_output_dir = self.output_base_dir / f"run_{run_id:02d}"
        run_output_dir.mkdir(exist_ok=True)
        
        # Build command
        cmd = [
            "python3", "main.py",
            "--config", config_file,
            "--output-dir", str(run_output_dir),
            "--duration", str(duration),
            "--agents", str(agents),
            "--contractors", str(contractors)
        ]
        
        # Set random seed if provided
        if seed is not None:
            # Modify config to include seed
            config = SimulationConfig.from_file(config_file)
            config.random_seed = seed
            temp_config_path = run_output_dir / "temp_config.yaml"
            config.save_to_file(str(temp_config_path))
            cmd[2] = str(temp_config_path)  # Use temp config
        
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
                performance_file = run_output_dir / "reports" / "performance_benchmarks.json"
                if performance_file.exists():
                    with open(performance_file, 'r') as f:
                        performance_data = json.load(f)
                    
                    # Extract key metrics
                    run_result = {
                        'run_id': run_id,
                        'config_file': config_file,
                        'seed': seed,
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
                        'output_dir': str(run_output_dir)
                    }
                    
                    # Calculate performance score
                    run_result['performance_score'] = self._calculate_performance_score(run_result)
                    
                else:
                    run_result = {
                        'run_id': run_id,
                        'success': False,
                        'error': 'No performance data generated',
                        'output_dir': str(run_output_dir)
                    }
            else:
                run_result = {
                    'run_id': run_id,
                    'success': False,
                    'error': result.stderr,
                    'output_dir': str(run_output_dir)
                }
                
        except subprocess.TimeoutExpired:
            run_result = {
                'run_id': run_id,
                'success': False,
                'error': 'Simulation timeout',
                'output_dir': str(run_output_dir)
            }
        except Exception as e:
            run_result = {
                'run_id': run_id,
                'success': False,
                'error': str(e),
                'output_dir': str(run_output_dir)
            }
        
        self.results.append(run_result)
        self.logger.info(f"Completed simulation run {run_id}: {'SUCCESS' if run_result['success'] else 'FAILED'}")
        
        return run_result
    
    def _calculate_performance_score(self, result):
        """Calculate overall performance score (0-100)."""
        if not result['success']:
            return 0
        
        # Weight different metrics
        cost_score = min(100, result['cost_reduction_pct'])
        time_score = min(100, result['time_savings_pct'])
        throughput_score = min(100, (result['throughput'] / 5.0) * 100)  # Normalize to 5 tasks/hour = 100%
        outsourcing_score = result['outsourcing_rate'] * 100
        
        # Weighted average
        performance_score = (
            cost_score * 0.3 +
            time_score * 0.3 +
            throughput_score * 0.2 +
            outsourcing_score * 0.2
        )
        
        return performance_score
    
    def run_batch(self, num_runs=20, config_scenarios=None):
        """Run multiple simulations with different configurations."""
        
        if config_scenarios is None:
            config_scenarios = [
                {
                    'config': 'config/default_config.yaml',
                    'duration': 7,
                    'agents': 15,
                    'contractors': 30,
                    'name': 'financial'
                },
                {
                    'config': 'config/content_creation_config.yaml',
                    'duration': 5,
                    'agents': 12,
                    'contractors': 25,
                    'name': 'content'
                },
                {
                    'config': 'config/customer_service_config.yaml',
                    'duration': 4,
                    'agents': 10,
                    'contractors': 20,
                    'name': 'service'
                }
            ]
        
        self.logger.info(f"Starting batch run of {num_runs} simulations")
        
        for run_id in range(1, num_runs + 1):
            # Cycle through different scenarios
            scenario = config_scenarios[(run_id - 1) % len(config_scenarios)]
            
            # Use different seeds to get variation
            seed = run_id * 42  # Different seed for each run
            
            self.run_single_simulation(
                run_id=run_id,
                config_file=scenario['config'],
                duration=scenario['duration'],
                agents=scenario['agents'],
                contractors=scenario['contractors'],
                seed=seed
            )
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate summary report of all runs."""
        
        # Save raw results
        results_file = self.output_base_dir / "batch_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create DataFrame for analysis
        successful_runs = [r for r in self.results if r['success']]
        failed_runs = [r for r in self.results if not r['success']]
        
        if successful_runs:
            df = pd.DataFrame(successful_runs)
            
            # Sort by performance score
            df_sorted = df.sort_values('performance_score')
            
            # Find best and worst performing runs
            worst_runs = df_sorted.head(2)
            best_runs = df_sorted.tail(2)
            
            # Generate report
            report_content = f"""# COALESCE Batch Simulation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Runs:** {len(self.results)}
**Successful Runs:** {len(successful_runs)}
**Failed Runs:** {len(failed_runs)}

## Summary Statistics

### Cost Reduction
- **Mean:** {df['cost_reduction_pct'].mean():.1f}%
- **Std Dev:** {df['cost_reduction_pct'].std():.1f}%
- **Min:** {df['cost_reduction_pct'].min():.1f}%
- **Max:** {df['cost_reduction_pct'].max():.1f}%

### Time Savings
- **Mean:** {df['time_savings_pct'].mean():.1f}%
- **Std Dev:** {df['time_savings_pct'].std():.1f}%
- **Min:** {df['time_savings_pct'].min():.1f}%
- **Max:** {df['time_savings_pct'].max():.1f}%

### System Throughput
- **Mean:** {df['throughput'].mean():.1f} tasks/hour
- **Std Dev:** {df['throughput'].std():.1f} tasks/hour
- **Min:** {df['throughput'].min():.1f} tasks/hour
- **Max:** {df['throughput'].max():.1f} tasks/hour

## Performance Analysis

### Paper Claims Validation
- **Runs achieving >50% cost reduction:** {len(df[df['cost_reduction_pct'] > 50])} / {len(df)} ({len(df[df['cost_reduction_pct'] > 50])/len(df)*100:.1f}%)
- **Runs achieving >80% time savings:** {len(df[df['time_savings_pct'] > 80])} / {len(df)} ({len(df[df['time_savings_pct'] > 80])/len(df)*100:.1f}%)
- **Runs with >50% outsourcing rate:** {len(df[df['outsourcing_rate'] > 0.5])} / {len(df)} ({len(df[df['outsourcing_rate'] > 0.5])/len(df)*100:.1f}%)

## Worst Performing Runs (Validation of Limitations)

"""
            
            for idx, (_, run) in enumerate(worst_runs.iterrows()):
                report_content += f"""### Worst Run #{idx + 1} (Run ID: {run['run_id']})
- **Performance Score:** {run['performance_score']:.1f}/100
- **Cost Reduction:** {run['cost_reduction_pct']:.1f}%
- **Time Savings:** {run['time_savings_pct']:.1f}%
- **Throughput:** {run['throughput']:.1f} tasks/hour
- **Outsourcing Rate:** {run['outsourcing_rate']*100:.1f}%
- **Configuration:** {run['config_file']}
- **Output Directory:** {run['output_dir']}

"""
            
            report_content += "## Best Performing Runs\n\n"
            
            for idx, (_, run) in enumerate(best_runs.iterrows()):
                report_content += f"""### Best Run #{idx + 1} (Run ID: {run['run_id']})
- **Performance Score:** {run['performance_score']:.1f}/100
- **Cost Reduction:** {run['cost_reduction_pct']:.1f}%
- **Time Savings:** {run['time_savings_pct']:.1f}%
- **Throughput:** {run['throughput']:.1f} tasks/hour
- **Outsourcing Rate:** {run['outsourcing_rate']*100:.1f}%
- **Configuration:** {run['config_file']}
- **Output Directory:** {run['output_dir']}

"""
            
            if failed_runs:
                report_content += f"## Failed Runs ({len(failed_runs)})\n\n"
                for run in failed_runs:
                    report_content += f"- **Run {run['run_id']}:** {run['error']}\n"
            
            # Save report
            report_file = self.output_base_dir / "batch_summary.md"
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            # Save CSV for further analysis
            csv_file = self.output_base_dir / "batch_results.csv"
            df.to_csv(csv_file, index=False)
            
            self.logger.info(f"Batch run completed. Summary saved to {report_file}")
            
            # Print key findings
            print("\n" + "="*80)
            print("BATCH SIMULATION RESULTS")
            print("="*80)
            print(f"Total Runs: {len(self.results)}")
            print(f"Successful: {len(successful_runs)}")
            print(f"Failed: {len(failed_runs)}")
            print(f"\nWorst Performance Score: {df['performance_score'].min():.1f}")
            print(f"Best Performance Score: {df['performance_score'].max():.1f}")
            print(f"Average Performance Score: {df['performance_score'].mean():.1f}")
            print("\nWorst performing runs found for validation:")
            for idx, (_, run) in enumerate(worst_runs.iterrows()):
                print(f"  Run {run['run_id']}: {run['performance_score']:.1f} score, {run['cost_reduction_pct']:.1f}% cost reduction")
            print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Run batch COALESCE simulations')
    parser.add_argument('--runs', type=int, default=20, help='Number of simulation runs')
    parser.add_argument('--output-dir', default='batch_results', help='Output directory for batch results')
    
    args = parser.parse_args()
    
    runner = BatchRunner(args.output_dir)
    runner.run_batch(num_runs=args.runs)


if __name__ == "__main__":
    main()