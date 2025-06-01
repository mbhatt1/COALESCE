#!/usr/bin/env python3
"""
Simple mathematical simulation runner that works reliably.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

def run_single_simulation(duration, agents, contractors):
    """Run a single simulation and return results."""
    cmd = [
        sys.executable, "main.py",
        "--duration", str(duration),
        "--agents", str(agents), 
        "--contractors", str(contractors)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            # Parse output for metrics
            output = result.stdout
            metrics = {}
            
            for line in output.split('\n'):
                if "Average Cost Reduction:" in line:
                    metrics['cost_reduction'] = float(line.split(':')[1].strip().replace('%', ''))
                elif "Average Time Savings:" in line:
                    metrics['time_savings'] = float(line.split(':')[1].strip().replace('%', ''))
                elif "Total Tasks Processed:" in line:
                    metrics['total_tasks'] = int(line.split(':')[1].strip().replace(',', ''))
                elif "System Throughput:" in line:
                    metrics['throughput'] = float(line.split(':')[1].strip().split()[0])
            
            return metrics
        else:
            print(f"Simulation failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """Run mathematical simulation experiments."""
    print("🔬 MATHEMATICAL SIMULATION")
    print("=" * 50)
    print("Running COALESCE theoretical validation")
    print("=" * 50)
    
    # Key experiment configurations
    experiments = [
        # Duration scaling
        (1, 15, 30), (7, 15, 30), (15, 15, 30), (30, 15, 30),
        # Agent scaling  
        (7, 5, 10), (7, 15, 30), (7, 25, 50), (7, 50, 100)
    ]
    
    all_results = []
    
    for i, (duration, agents, contractors) in enumerate(experiments, 1):
        print(f"\n📊 Experiment {i}/{len(experiments)}: {duration}d, {agents}a, {contractors}c")
        
        # Run 5 times for each configuration
        exp_results = []
        for run in range(5):
            print(f"  Run {run + 1}/5", end="")
            result = run_single_simulation(duration, agents, contractors)
            if result:
                result.update({
                    'experiment': i,
                    'duration': duration,
                    'agents': agents,
                    'contractors': contractors,
                    'run': run + 1
                })
                exp_results.append(result)
                print(f" ✅ Cost: {result.get('cost_reduction', 0):.1f}%")
            else:
                print(" ❌ Failed")
        
        all_results.extend(exp_results)
    
    # Calculate aggregate statistics
    if all_results:
        cost_reductions = [r['cost_reduction'] for r in all_results if 'cost_reduction' in r]
        time_savings = [r['time_savings'] for r in all_results if 'time_savings' in r]
        
        avg_cost = sum(cost_reductions) / len(cost_reductions) if cost_reductions else 0
        avg_time = sum(time_savings) / len(time_savings) if time_savings else 0
        
        print("\n" + "=" * 50)
        print("📈 MATHEMATICAL SIMULATION RESULTS")
        print("=" * 50)
        print(f"Total Successful Runs: {len(all_results)}")
        print(f"Average Cost Reduction: {avg_cost:.1f}%")
        print(f"Average Time Savings: {avg_time:.1f}%")
        print("=" * 50)
        
        # Save results
        output_file = f"mathematical_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_runs': len(all_results),
                    'avg_cost_reduction': avg_cost,
                    'avg_time_savings': avg_time,
                    'timestamp': datetime.now().isoformat()
                },
                'detailed_results': all_results
            }, f, indent=2)
        
        print(f"📁 Results saved to: {output_file}")
        
        return {
            'avg_cost_reduction': avg_cost,
            'avg_time_savings': avg_time,
            'total_runs': len(all_results)
        }
    else:
        print("❌ No successful runs!")
        return None

if __name__ == "__main__":
    main()