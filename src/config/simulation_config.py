"""
Configuration management for COALESCE simulation.

This module handles loading and managing simulation configuration parameters.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.constants import *


@dataclass
class SimulationConfig:
    """Configuration parameters for COALESCE simulation."""
    
    # Simulation duration and timing
    simulation_duration_days: int = DEFAULT_SIMULATION_DURATION_DAYS
    time_step_seconds: int = DEFAULT_TIME_STEP_SECONDS
    
    # Agent configuration
    num_client_agents: int = DEFAULT_NUM_CLIENT_AGENTS
    num_contractor_agents: int = DEFAULT_NUM_CONTRACTOR_AGENTS
    
    # Task generation parameters
    poisson_lambda: float = DEFAULT_POISSON_LAMBDA
    task_type_distribution: Dict[str, float] = None
    complexity_factor_range: tuple = COMPLEXITY_FACTOR_RANGE
    value_range: tuple = VALUE_RANGE
    data_size_range_gb: tuple = DATA_SIZE_RANGE_GB
    
    # Market dynamics
    business_hours_multiplier: float = BUSINESS_HOURS_MULTIPLIER
    quarterly_earnings_multiplier: float = QUARTERLY_EARNINGS_MULTIPLIER
    maintenance_capacity_reduction: float = MAINTENANCE_CAPACITY_REDUCTION
    
    # Decision algorithm parameters
    skill_compatibility_threshold: float = SKILL_COMPATIBILITY_THRESHOLD
    topsis_threshold: float = TOPSIS_THRESHOLD
    min_confidence_threshold: float = MIN_CONFIDENCE_THRESHOLD
    learning_rate: float = LEARNING_RATE
    
    # Cost model parameters
    energy_cost_per_kwh: float = ENERGY_COST_PER_KWH
    data_transfer_cost_per_gb: float = DATA_TRANSFER_COST_PER_GB
    protocol_overhead_base_cost: float = PROTOCOL_OVERHEAD_BASE_COST
    integration_base_cost: float = INTEGRATION_BASE_COST
    latency_penalty_per_minute: float = LATENCY_PENALTY_PER_MINUTE
    
    # Security parameters
    security_overhead_enabled: bool = True
    chacha20_overhead_percent: float = CHACHA20_OVERHEAD_PERCENT
    sgx_enclave_overhead_percent: float = SGX_ENCLAVE_OVERHEAD_PERCENT
    
    # Network parameters
    default_bandwidth_mbps: int = DEFAULT_BANDWIDTH_MBPS
    default_latency_ms: float = DEFAULT_LATENCY_MS
    default_reliability: float = DEFAULT_RELIABILITY
    
    # Contractor distribution
    contractor_type_distribution: Dict[str, float] = None
    
    # Variation parameters
    latency_variation_percent: float = LATENCY_VARIATION_PERCENT
    cost_variation_percent: float = COST_VARIATION_PERCENT
    demand_variation_percent: float = DEMAND_VARIATION_PERCENT
    
    # Output configuration
    enable_detailed_logging: bool = True
    enable_time_series_tracking: bool = True
    enable_agent_performance_tracking: bool = True
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Initialize default values for complex fields."""
        if self.task_type_distribution is None:
            self.task_type_distribution = TASK_TYPE_DISTRIBUTION.copy()
        
        if self.contractor_type_distribution is None:
            self.contractor_type_distribution = CONTRACTOR_TYPE_DISTRIBUTION.copy()
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SimulationConfig':
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Create default config file
            default_config = cls()
            default_config.save_to_file(config_path)
            return default_config
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create config object with loaded data
        config = cls()
        
        # Update fields from loaded data
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            config_dict[field_name] = value
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            result[field_name] = value
        return result
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        # Check positive values
        if self.simulation_duration_days <= 0:
            errors.append("simulation_duration_days must be positive")
        
        if self.num_client_agents <= 0:
            errors.append("num_client_agents must be positive")
        
        if self.num_contractor_agents <= 0:
            errors.append("num_contractor_agents must be positive")
        
        # Check probability distributions
        if abs(sum(self.task_type_distribution.values()) - 1.0) > 0.01:
            errors.append("task_type_distribution must sum to 1.0")
        
        if abs(sum(self.contractor_type_distribution.values()) - 1.0) > 0.01:
            errors.append("contractor_type_distribution must sum to 1.0")
        
        # Check threshold values
        if not 0 <= self.skill_compatibility_threshold <= 1:
            errors.append("skill_compatibility_threshold must be between 0 and 1")
        
        if not 0 <= self.topsis_threshold <= 1:
            errors.append("topsis_threshold must be between 0 and 1")
        
        if not 0 <= self.min_confidence_threshold <= 1:
            errors.append("min_confidence_threshold must be between 0 and 1")
        
        # Check ranges
        if self.complexity_factor_range[0] >= self.complexity_factor_range[1]:
            errors.append("complexity_factor_range must be a valid range")
        
        if self.value_range[0] >= self.value_range[1]:
            errors.append("value_range must be a valid range")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True


def create_default_config_file(config_path: str = "config/default_config.yaml"):
    """Create a default configuration file with documentation."""
    config = SimulationConfig()
    
    # Create the directory if it doesn't exist
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create YAML content with comments
    yaml_content = f"""# COALESCE Simulation Configuration
# This file contains all configuration parameters for the COALESCE simulation

# Simulation Duration and Timing
simulation_duration_days: {config.simulation_duration_days}  # Number of days to simulate
time_step_seconds: {config.time_step_seconds}  # Time step in seconds (3600 = 1 hour)

# Agent Configuration
num_client_agents: {config.num_client_agents}  # Number of client agents
num_contractor_agents: {config.num_contractor_agents}  # Number of contractor agents

# Task Generation Parameters
poisson_lambda: {config.poisson_lambda}  # Poisson arrival rate (tasks per minute during peak)

# Task type distribution (must sum to 1.0)
task_type_distribution:
  financial_rag: {config.task_type_distribution['financial_rag']}
  risk_assessment: {config.task_type_distribution['risk_assessment']}
  sentiment_analysis: {config.task_type_distribution['sentiment_analysis']}
  portfolio_optimization: {config.task_type_distribution['portfolio_optimization']}

# Task parameter ranges
complexity_factor_range: [{config.complexity_factor_range[0]}, {config.complexity_factor_range[1]}]
value_range: [{config.value_range[0]}, {config.value_range[1]}]
data_size_range_gb: [{config.data_size_range_gb[0]}, {config.data_size_range_gb[1]}]

# Market Dynamics
business_hours_multiplier: {config.business_hours_multiplier}  # Demand multiplier during business hours
quarterly_earnings_multiplier: {config.quarterly_earnings_multiplier}  # Seasonal demand increase
maintenance_capacity_reduction: {config.maintenance_capacity_reduction}  # Capacity reduction during maintenance

# Decision Algorithm Parameters
skill_compatibility_threshold: {config.skill_compatibility_threshold}  # θ_skill from paper
topsis_threshold: {config.topsis_threshold}  # τ_threshold from paper
min_confidence_threshold: {config.min_confidence_threshold}  # ρ_min from paper
learning_rate: {config.learning_rate}  # η for weight updates

# Cost Model Parameters
energy_cost_per_kwh: {config.energy_cost_per_kwh}  # USD per kWh
data_transfer_cost_per_gb: {config.data_transfer_cost_per_gb}  # USD per GB
protocol_overhead_base_cost: {config.protocol_overhead_base_cost}  # USD base protocol cost
integration_base_cost: {config.integration_base_cost}  # USD integration cost
latency_penalty_per_minute: {config.latency_penalty_per_minute}  # USD penalty per minute

# Security Parameters
security_overhead_enabled: {config.security_overhead_enabled}
chacha20_overhead_percent: {config.chacha20_overhead_percent}  # Encryption overhead
sgx_enclave_overhead_percent: {config.sgx_enclave_overhead_percent}  # SGX overhead

# Network Parameters
default_bandwidth_mbps: {config.default_bandwidth_mbps}
default_latency_ms: {config.default_latency_ms}
default_reliability: {config.default_reliability}

# Contractor Type Distribution (must sum to 1.0)
contractor_type_distribution:
  contractor_b: {config.contractor_type_distribution['contractor_b']}  # Premium GPU Specialist
  contractor_c: {config.contractor_type_distribution['contractor_c']}  # High-Throughput CPU
  contractor_d: {config.contractor_type_distribution['contractor_d']}  # Budget Provider
  contractor_e: {config.contractor_type_distribution['contractor_e']}  # Edge Specialist
  contractor_f: {config.contractor_type_distribution['contractor_f']}  # Hybrid Cloud
  contractor_g: {config.contractor_type_distribution['contractor_g']}  # Quantum-Enhanced

# Variation Parameters (for realistic simulation)
latency_variation_percent: {config.latency_variation_percent}  # ±15% latency variation
cost_variation_percent: {config.cost_variation_percent}  # ±10% cost variation
demand_variation_percent: {config.demand_variation_percent}  # ±25% demand variation

# Output Configuration
enable_detailed_logging: {config.enable_detailed_logging}
enable_time_series_tracking: {config.enable_time_series_tracking}
enable_agent_performance_tracking: {config.enable_agent_performance_tracking}

# Random seed for reproducibility (null for random)
random_seed: {config.random_seed}
"""
    
    with open(config_file, 'w') as f:
        f.write(yaml_content)
    
    return config_file


if __name__ == "__main__":
    # Create default configuration file
    config_file = create_default_config_file()
    print(f"Created default configuration file: {config_file}")