#!/usr/bin/env python3
"""
COALESCE: Cost-Optimized Agent Labor Exchange via Skill-based Competence Estimation
Main simulation runner and report generator

This module implements the complete COALESCE framework as described in the research paper,
including cost modeling, decision algorithms, and comprehensive simulation.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from src.simulation.simulation_engine import COALESCESimulation
from src.reporting.report_generator import ReportGenerator
from src.config.simulation_config import SimulationConfig
from src.utils.logger import setup_logging


def main():
    """Main entry point for COALESCE simulation."""
    parser = argparse.ArgumentParser(
        description="COALESCE: Cost-Optimized Agent Labor Exchange Simulation"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default_config.yaml",
        help="Path to simulation configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Directory for simulation output and reports"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=30,
        help="Simulation duration in days"
    )
    parser.add_argument(
        "--agents", 
        type=int, 
        default=50,
        help="Number of client agents"
    )
    parser.add_argument(
        "--contractors", 
        type=int, 
        default=200,
        help="Number of contractor agents"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("Starting COALESCE simulation...")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Duration: {args.duration} days")
        logger.info(f"Agents: {args.agents} clients, {args.contractors} contractors")
        
        # Load configuration
        config = SimulationConfig.from_file(args.config)
        config.simulation_duration_days = args.duration
        config.num_client_agents = args.agents
        config.num_contractor_agents = args.contractors
        
        # Initialize simulation
        simulation = COALESCESimulation(config)
        
        # Run simulation
        logger.info("Running simulation...")
        results = simulation.run()
        
        # Generate comprehensive report
        logger.info("Generating reports...")
        report_generator = ReportGenerator(results, config, output_dir)
        report_paths = report_generator.generate_all_reports()
        
        logger.info("Simulation completed successfully!")
        logger.info("Generated reports:")
        for report_type, path in report_paths.items():
            logger.info(f"  {report_type}: {path}")
            
        # Print summary to console
        print("\n" + "="*80)
        print("COALESCE SIMULATION SUMMARY")
        print("="*80)
        print(f"Simulation Duration: {args.duration} days")
        print(f"Total Tasks Processed: {results.total_tasks:,}")
        print(f"Average Cost Reduction: {results.avg_cost_reduction:.1%}")
        print(f"Average Time Savings: {results.avg_time_savings:.1%}")
        print(f"System Throughput: {results.system_throughput:.1f} tasks/hour")
        print(f"Energy Efficiency Gain: {results.energy_efficiency_gain:.1%}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()