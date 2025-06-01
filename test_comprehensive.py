#!/usr/bin/env python3
"""
Test version of comprehensive analysis with just 5 runs per experiment.
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

def main():
    """Test with just a few experiments."""
    
    logger.info("Testing comprehensive analysis with 5 runs each...")
    
    # Test configurations
    test_configs = [
        (1, 5, 10),   # dur_01
        (7, 5, 10),   # agt_01  
        (7, 15, 30),  # agt_03
    ]
    
    all_results = []
    
    for i, (duration, agents, contractors) in enumerate(test_configs):
        exp_id = f'test_{i+1:02d}'
        logger.info(f"Test experiment {exp_id}: {duration}d, {agents}a, {contractors}c")
        
        for run in range(5):
            logger.info(f"  Run {run + 1}/5")
            
            result = run_single_experiment(duration, agents, contractors, run + 1)
            if result:
                result['experiment_id'] = exp_id
                all_results.append(result)
    
    # Analyze results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Calculate statistics
        experiment_stats = df.groupby('experiment_id').agg({
            'cost_reduction': ['mean', 'std'],
            'time_savings': ['mean', 'std'],
            'outsourcing_rate': ['mean', 'std'],
            'topsis_score': ['mean', 'std'],
            'total_tasks': 'mean',
            'duration': 'first',
            'agents': 'first'
        }).round(1)
        
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Total successful runs: {len(df)}")
        print("\nPer-experiment statistics:")
        print(experiment_stats)
        
        # Save test results
        output_dir = Path("data/test_comprehensive")
        output_dir.mkdir(exist_ok=True)
        df.to_csv(output_dir / "test_results.csv", index=False)
        experiment_stats.to_csv(output_dir / "test_statistics.csv")
        
        logger.info(f"Test results saved to {output_dir}")
        
        return True
    else:
        logger.error("No successful test runs!")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Test successful! Ready to run full comprehensive analysis.")
    else:
        print("Test failed! Check configuration.")