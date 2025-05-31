"""
COALESCE Simulation Engine

This module implements the complete simulation framework for the COALESCE system,
including agent behavior, market dynamics, and performance measurement.
"""

import logging
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from ..agents.agent_types import ClientAgent, ContractorAgent, Task, create_contractor_from_paper_spec, create_sample_task
from ..decision.decision_engine import DecisionEngine, DecisionResult
from ..cost_model.cost_calculator import CostCalculator
from ..utils.constants import *


@dataclass
class SimulationMetrics:
    """Container for simulation performance metrics."""
    total_tasks: int = 0
    outsourced_tasks: int = 0
    local_tasks: int = 0
    
    total_cost_local: float = 0.0
    total_cost_outsourced: float = 0.0
    total_cost_savings: float = 0.0
    
    total_time_local: float = 0.0
    total_time_outsourced: float = 0.0
    total_time_savings: float = 0.0
    
    avg_topsis_score: float = 0.0
    avg_confidence: float = 0.0
    
    system_throughput: float = 0.0
    energy_efficiency_gain: float = 0.0
    
    contractor_utilization: Dict[str, float] = field(default_factory=dict)
    market_concentration_hhi: float = 0.0
    
    @property
    def outsourcing_rate(self) -> float:
        return self.outsourced_tasks / self.total_tasks if self.total_tasks > 0 else 0.0
    
    @property
    def avg_cost_reduction(self) -> float:
        if self.total_cost_local > 0:
            return self.total_cost_savings / self.total_cost_local
        return 0.0
    
    @property
    def avg_time_savings(self) -> float:
        if self.total_time_local > 0:
            return self.total_time_savings / self.total_time_local
        return 0.0


@dataclass
class SimulationResults:
    """Complete simulation results with detailed metrics and time series data."""
    metrics: SimulationMetrics
    time_series: Dict[str, List[float]] = field(default_factory=dict)
    decision_history: List[DecisionResult] = field(default_factory=list)
    agent_performance: Dict[str, Dict] = field(default_factory=dict)
    market_dynamics: Dict[str, List[float]] = field(default_factory=dict)
    
    # Convenience properties for main.py
    @property
    def total_tasks(self) -> int:
        return self.metrics.total_tasks
    
    @property
    def avg_cost_reduction(self) -> float:
        return self.metrics.avg_cost_reduction
    
    @property
    def avg_time_savings(self) -> float:
        return self.metrics.avg_time_savings
    
    @property
    def system_throughput(self) -> float:
        return self.metrics.system_throughput
    
    @property
    def energy_efficiency_gain(self) -> float:
        return self.metrics.energy_efficiency_gain


class COALESCESimulation:
    """
    Main simulation engine for the COALESCE framework.
    
    Implements the complete simulation as described in Section 4 of the paper,
    including agent behavior, market dynamics, and performance measurement.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.decision_engine = DecisionEngine()
        self.cost_calculator = CostCalculator()
        
        # Initialize agents
        self.client_agents: List[ClientAgent] = []
        self.contractor_agents: List[ContractorAgent] = []
        
        # Simulation state
        self.current_time = datetime.now()
        self.simulation_step = 0
        self.metrics = SimulationMetrics()
        
        # Time series tracking
        self.time_series = defaultdict(list)
        self.decision_history = []
        
        # Market state
        self.market_demand = 0.5
        self.market_supply = 0.8
        
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize client and contractor agents based on configuration."""
        self.logger.info("Initializing simulation agents...")
        
        # Create client agents
        for i in range(self.config.num_client_agents):
            client = ClientAgent(name=f"Client_{i:03d}")
            self.client_agents.append(client)
        
        # Create contractor agents based on paper specifications
        contractor_types = list(CONTRACTOR_TYPE_DISTRIBUTION.keys())
        contractor_weights = list(CONTRACTOR_TYPE_DISTRIBUTION.values())
        
        for i in range(self.config.num_contractor_agents):
            # Select contractor type based on distribution from paper
            contractor_type = np.random.choice(contractor_types, p=contractor_weights)
            contractor = create_contractor_from_paper_spec(contractor_type)
            contractor.name = f"{contractor.name}_{i:03d}"
            
            # Add some variation to base parameters
            self._add_contractor_variation(contractor)
            
            self.contractor_agents.append(contractor)
        
        self.logger.info(f"Created {len(self.client_agents)} client agents and {len(self.contractor_agents)} contractor agents")
    
    def _add_contractor_variation(self, contractor: ContractorAgent):
        """Add realistic variation to contractor parameters."""
        # Add latency variation
        latency_variation = np.random.normal(0, contractor.avg_latency_minutes * LATENCY_VARIATION_PERCENT)
        contractor.avg_latency_minutes = max(1.0, contractor.avg_latency_minutes + latency_variation)
        
        # Add cost variation
        cost_variation = np.random.normal(0, contractor.base_price_per_task * COST_VARIATION_PERCENT)
        contractor.base_price_per_task = max(1.0, contractor.base_price_per_task + cost_variation)
        
        # Add demand variation
        contractor.current_demand = np.random.uniform(0.2, 0.9)
        contractor.available_capacity = np.random.uniform(0.6, 1.0)
    
    def run(self) -> SimulationResults:
        """
        Run the complete COALESCE simulation.
        
        Returns comprehensive results including metrics, time series, and analysis.
        """
        self.logger.info(f"Starting COALESCE simulation for {self.config.simulation_duration_days} days")
        
        total_steps = int(self.config.simulation_duration_days * 24)  # Hourly steps
        
        for step in range(total_steps):
            self.simulation_step = step
            self.current_time = datetime.now() + timedelta(hours=step)
            
            # Update market conditions
            self._update_market_conditions()
            
            # Generate new tasks
            new_tasks = self._generate_tasks()
            
            # Distribute tasks to client agents
            self._distribute_tasks(new_tasks)
            
            # Process decisions for all client agents
            self._process_agent_decisions()
            
            # Update contractor states
            self._update_contractor_states()
            
            # Record metrics
            self._record_metrics()
            
            if step % 24 == 0:  # Log daily progress
                self.logger.info(f"Simulation day {step // 24 + 1}/{self.config.simulation_duration_days} completed")
        
        # Calculate final metrics
        self._calculate_final_metrics()
        
        # Create results object
        results = SimulationResults(
            metrics=self.metrics,
            time_series=dict(self.time_series),
            decision_history=self.decision_history,
            agent_performance=self._calculate_agent_performance(),
            market_dynamics=self._calculate_market_dynamics()
        )
        
        self.logger.info("Simulation completed successfully")
        return results
    
    def _update_market_conditions(self):
        """Update market supply and demand based on time and external factors."""
        hour_of_day = self.current_time.hour
        day_of_week = self.current_time.weekday()
        
        # Business hours effect (9 AM - 5 PM weekdays)
        if 9 <= hour_of_day <= 17 and day_of_week < 5:
            demand_multiplier = BUSINESS_HOURS_MULTIPLIER
        else:
            demand_multiplier = 1.0
        
        # Add some random variation
        demand_noise = np.random.normal(0, 0.1)
        supply_noise = np.random.normal(0, 0.05)
        
        self.market_demand = np.clip(0.5 * demand_multiplier + demand_noise, 0.1, 2.0)
        self.market_supply = np.clip(0.8 + supply_noise, 0.3, 1.0)
        
        # Update contractor demand based on market conditions
        for contractor in self.contractor_agents:
            base_demand = contractor.current_demand
            market_effect = (self.market_demand - 0.5) * 0.3
            contractor.current_demand = np.clip(base_demand + market_effect, 0.1, 1.0)
    
    def _generate_tasks(self) -> List[Task]:
        """Generate new tasks based on Poisson arrival process."""
        # Poisson arrival rate adjusted for market demand
        lambda_rate = DEFAULT_POISSON_LAMBDA * self.market_demand / 60  # Convert to per-minute rate
        
        # Number of tasks this hour
        num_tasks = np.random.poisson(lambda_rate * 60)  # 60 minutes per hour
        
        tasks = []
        task_types = list(TASK_TYPE_DISTRIBUTION.keys())
        task_weights = list(TASK_TYPE_DISTRIBUTION.values())
        
        for _ in range(num_tasks):
            # Select task type based on distribution
            task_type = np.random.choice(task_types, p=task_weights)
            
            # Create task with some variation
            task = create_sample_task(task_type)
            
            # Add variation to task parameters
            task.complexity_factor = np.random.uniform(*COMPLEXITY_FACTOR_RANGE)
            task.value = np.random.uniform(*VALUE_RANGE)
            task.data_size_gb = np.random.uniform(*DATA_SIZE_RANGE_GB)
            task.budget_constraint = task.value * np.random.uniform(0.3, 0.8)
            
            tasks.append(task)
        
        return tasks
    
    def _distribute_tasks(self, tasks: List[Task]):
        """Distribute tasks to client agents."""
        if not tasks:
            return
        
        # Simple round-robin distribution
        for i, task in enumerate(tasks):
            client_idx = i % len(self.client_agents)
            self.client_agents[client_idx].task_queue.append(task)
    
    def _process_agent_decisions(self):
        """Process decision-making for all client agents."""
        for client in self.client_agents:
            while client.task_queue:
                task = client.task_queue.pop(0)
                
                # Make decision using COALESCE algorithm
                decision = self.decision_engine.make_decision(
                    client, task, self.contractor_agents
                )
                
                # Execute decision
                self._execute_decision(client, task, decision)
                
                # Record decision
                self.decision_history.append(decision)
    
    def _execute_decision(self, client: ClientAgent, task: Task, decision: DecisionResult):
        """Execute the decision (local or outsourced execution)."""
        self.metrics.total_tasks += 1
        
        if decision.decision == 'LOCAL':
            self._execute_local_task(client, task, decision)
        else:
            self._execute_outsourced_task(client, task, decision)
        
        # Update client statistics
        client.completed_tasks.append(task)
        client.total_cost_spent += decision.criteria_scores.cost
        
        if decision.decision == 'OUTSOURCE':
            client.total_time_saved += decision.time_savings
            client.outsourcing_rate = len([d for d in self.decision_history 
                                         if d.decision == 'OUTSOURCE']) / len(self.decision_history)
    
    def _execute_local_task(self, client: ClientAgent, task: Task, decision: DecisionResult):
        """Execute task locally."""
        self.metrics.local_tasks += 1
        
        # Calculate actual costs (with some variation)
        cost_components = self.cost_calculator.calculate_internal_cost(client, task)
        actual_cost = cost_components.internal_total * np.random.normal(1.0, 0.05)
        
        self.metrics.total_cost_local += actual_cost
        
        # Estimate execution time
        estimated_time = self._estimate_local_execution_time(task)
        self.metrics.total_time_local += estimated_time
        
        # Update cost calibration
        self.cost_calculator.calibrate_costs(
            client.agent_id, decision.criteria_scores.cost, actual_cost
        )
    
    def _execute_outsourced_task(self, client: ClientAgent, task: Task, decision: DecisionResult):
        """Execute task via outsourcing."""
        self.metrics.outsourced_tasks += 1
        contractor = decision.selected_contractor
        
        # Calculate actual costs (with some variation)
        cost_components = self.cost_calculator.calculate_external_cost(contractor, task, client)
        actual_cost = cost_components.external_total * np.random.normal(1.0, 0.08)
        
        self.metrics.total_cost_outsourced += actual_cost
        
        # Actual execution time (contractor's latency + some variation)
        actual_time = contractor.avg_latency_minutes * np.random.normal(1.0, 0.1)
        self.metrics.total_time_outsourced += actual_time
        
        # Calculate savings
        estimated_local_cost = self.cost_calculator.calculate_internal_cost(client, task).internal_total
        estimated_local_time = self._estimate_local_execution_time(task)
        
        cost_savings = estimated_local_cost - actual_cost
        time_savings = estimated_local_time - actual_time
        
        self.metrics.total_cost_savings += max(0, cost_savings)
        self.metrics.total_time_savings += max(0, time_savings)
        
        # Update contractor statistics
        contractor.completed_tasks += 1
        contractor.total_revenue += actual_cost
        
        # Update cost calibration
        self.cost_calculator.calibrate_costs(
            contractor.agent_id, decision.criteria_scores.cost, actual_cost
        )
    
    def _estimate_local_execution_time(self, task: Task) -> float:
        """Estimate local execution time in minutes."""
        base_times = {
            'financial_rag': 125.4,
            'risk_assessment': 90.0,
            'sentiment_analysis': 45.0,
            'portfolio_optimization': 180.0
        }
        
        base_time = base_times.get(task.task_type, 100.0)
        return base_time * task.complexity_factor
    
    def _update_contractor_states(self):
        """Update contractor availability and market position."""
        for contractor in self.contractor_agents:
            # Update capacity based on completed tasks
            if contractor.completed_tasks > 0:
                utilization = min(1.0, contractor.completed_tasks / 10.0)  # Assume max 10 tasks per hour
                contractor.available_capacity = max(0.1, 1.0 - utilization)
            
            # Update market share based on performance
            if self.metrics.total_tasks > 0:
                contractor.market_share = contractor.completed_tasks / self.metrics.total_tasks
    
    def _record_metrics(self):
        """Record time series metrics."""
        current_hour = self.simulation_step
        
        # Record basic metrics
        self.time_series['total_tasks'].append(self.metrics.total_tasks)
        self.time_series['outsourcing_rate'].append(self.metrics.outsourcing_rate)
        self.time_series['market_demand'].append(self.market_demand)
        self.time_series['market_supply'].append(self.market_supply)
        
        # Record cost metrics
        if self.metrics.total_cost_local > 0:
            self.time_series['cost_reduction'].append(self.metrics.avg_cost_reduction)
        else:
            self.time_series['cost_reduction'].append(0.0)
        
        # Record time metrics
        if self.metrics.total_time_local > 0:
            self.time_series['time_savings'].append(self.metrics.avg_time_savings)
        else:
            self.time_series['time_savings'].append(0.0)
        
        # Record decision quality metrics
        recent_decisions = self.decision_history[-10:] if len(self.decision_history) >= 10 else self.decision_history
        if recent_decisions:
            outsource_decisions = [d.topsis_score for d in recent_decisions if d.decision == 'OUTSOURCE']
            avg_topsis = np.mean(outsource_decisions) if outsource_decisions else 0.0
            avg_confidence = np.mean([d.confidence for d in recent_decisions]) if recent_decisions else 0.0
            self.time_series['avg_topsis_score'].append(avg_topsis if not np.isnan(avg_topsis) else 0.0)
            self.time_series['avg_confidence'].append(avg_confidence)
        else:
            self.time_series['avg_topsis_score'].append(0.0)
            self.time_series['avg_confidence'].append(0.0)
    
    def _calculate_final_metrics(self):
        """Calculate final simulation metrics."""
        # Calculate average TOPSIS score and confidence
        outsourced_decisions = [d for d in self.decision_history if d.decision == 'OUTSOURCE']
        
        if outsourced_decisions:
            self.metrics.avg_topsis_score = np.mean([d.topsis_score for d in outsourced_decisions])
            self.metrics.avg_confidence = np.mean([d.confidence for d in self.decision_history]) if self.decision_history else 0.0
        
        # Calculate system throughput (tasks per hour)
        total_hours = self.config.simulation_duration_days * 24
        self.metrics.system_throughput = self.metrics.total_tasks / total_hours
        
        # Calculate energy efficiency gain
        if self.metrics.total_time_local > 0 and self.metrics.total_time_outsourced > 0:
            local_efficiency = DOCUMENTS_PER_KWH_LOCAL
            outsourced_efficiency = DOCUMENTS_PER_KWH_OPTIMIZED
            self.metrics.energy_efficiency_gain = (outsourced_efficiency - local_efficiency) / local_efficiency
        
        # Calculate contractor utilization
        for contractor in self.contractor_agents:
            if contractor.completed_tasks > 0:
                self.metrics.contractor_utilization[contractor.name] = contractor.completed_tasks / self.metrics.total_tasks
        
        # Calculate market concentration (HHI)
        market_shares = [contractor.market_share for contractor in self.contractor_agents]
        self.metrics.market_concentration_hhi = sum(share ** 2 for share in market_shares) * 10000
    
    def _calculate_agent_performance(self) -> Dict[str, Dict]:
        """Calculate detailed agent performance metrics."""
        performance = {}
        
        # Client agent performance
        for client in self.client_agents:
            performance[client.name] = {
                'completed_tasks': len(client.completed_tasks),
                'total_cost_spent': client.total_cost_spent,
                'total_time_saved': client.total_time_saved,
                'outsourcing_rate': client.outsourcing_rate
            }
        
        # Contractor agent performance
        for contractor in self.contractor_agents:
            performance[contractor.name] = {
                'completed_tasks': contractor.completed_tasks,
                'total_revenue': contractor.total_revenue,
                'market_share': contractor.market_share,
                'avg_latency': contractor.avg_latency_minutes,
                'reliability_score': contractor.reliability_score,
                'specialization': contractor.specialization
            }
        
        return performance
    
    def _calculate_market_dynamics(self) -> Dict[str, List[float]]:
        """Calculate market dynamics data."""
        return {
            'demand_history': self.time_series.get('market_demand', []),
            'supply_history': self.time_series.get('market_supply', []),
            'price_volatility': self._calculate_price_volatility(),
            'concentration_evolution': self._calculate_concentration_evolution()
        }
    
    def _calculate_price_volatility(self) -> List[float]:
        """Calculate price volatility over time."""
        volatility = []
        window_size = 24  # 24-hour window
        
        for i in range(len(self.time_series.get('market_demand', []))):
            if i < window_size:
                volatility.append(0.1)  # Default low volatility
            else:
                demand_window = self.time_series['market_demand'][i-window_size:i]
                mean_demand = np.mean(demand_window) if demand_window else 0.1
                cv = np.std(demand_window) / mean_demand if mean_demand > 0 else 0.1
                volatility.append(cv)
        
        return volatility
    
    def _calculate_concentration_evolution(self) -> List[float]:
        """Calculate market concentration evolution over time."""
        # Simplified calculation - would need more detailed tracking in a real implementation
        return [self.metrics.market_concentration_hhi] * len(self.time_series.get('total_tasks', []))