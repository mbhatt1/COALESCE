"""
Cost calculation module implementing the comprehensive cost model from the COALESCE paper.

This module implements the detailed cost formulations from Equations (1)-(8) in the paper,
including internal costs (compute, memory, energy, opportunity, depreciation) and 
external costs (pricing, communication, verification, integration, risk, latency penalties).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from ..agents.agent_types import ClientAgent, ContractorAgent, Task
from ..utils.constants import *


@dataclass
class CostComponents:
    """Container for detailed cost breakdown."""
    compute_cost: float = 0.0
    memory_cost: float = 0.0
    energy_cost: float = 0.0
    opportunity_cost: float = 0.0
    depreciation_cost: float = 0.0
    communication_cost: float = 0.0
    verification_cost: float = 0.0
    integration_cost: float = 0.0
    risk_cost: float = 0.0
    latency_penalty: float = 0.0
    base_price: float = 0.0
    
    @property
    def internal_total(self) -> float:
        """Total internal execution cost."""
        return (self.compute_cost + self.memory_cost + self.energy_cost + 
                self.opportunity_cost + self.depreciation_cost)
    
    @property
    def external_total(self) -> float:
        """Total external outsourcing cost."""
        return (self.base_price + self.communication_cost + self.verification_cost +
                self.integration_cost + self.risk_cost + self.latency_penalty)


class CostCalculator:
    """
    Implements the comprehensive cost model from the COALESCE paper.
    
    Based on Equations (1)-(8) from Section 3.5 Cost Modeling.
    """
    
    def __init__(self):
        self.calibration_history: Dict[str, List[float]] = {}
        self.ewma_lambda = 0.2  # Learning rate for cost calibration
    
    def calculate_internal_cost(self, agent: ClientAgent, task: Task) -> CostComponents:
        """
        Calculate internal execution cost using Eq. (1) from the paper.
        
        C_internal(T) = C_compute + C_memory + C_energy + C_opportunity + C_depreciation
        """
        components = CostComponents()
        
        # Compute Cost (Eq. 2): Based on FLOPS estimation
        flops_required = self._estimate_flops(task)
        execution_time_hours = flops_required / (agent.hardware.peak_flops * agent.hardware.utilization_factor)
        components.compute_cost = execution_time_hours * agent.hardware.compute_cost_per_hour
        
        # Memory Cost (Eq. 3): VRAM and system memory utilization
        memory_usage = self._calculate_memory_usage(task, agent)
        memory_utilization = memory_usage / agent.hardware.total_memory_gb
        components.memory_cost = (memory_utilization * execution_time_hours * 
                                agent.hardware.memory_cost_per_hour)
        
        # Energy Cost (Eq. 4): Power consumption
        power_consumption_kw = (agent.hardware.tdp_watts * agent.hardware.utilization_factor) / 1000
        components.energy_cost = power_consumption_kw * execution_time_hours * ENERGY_COST_PER_KWH
        
        # Opportunity Cost (Eq. 5): Value of alternative tasks
        components.opportunity_cost = self._calculate_opportunity_cost(agent, execution_time_hours)
        
        # Depreciation Cost: Usage-based hardware depreciation
        components.depreciation_cost = self._calculate_depreciation_cost(agent, execution_time_hours)
        
        return components
    
    def calculate_external_cost(self, contractor: ContractorAgent, task: Task, 
                              client: ClientAgent) -> CostComponents:
        """
        Calculate external outsourcing cost using Eq. (6) from the paper.
        
        C_external(T, A_j) = P_j(T) + C_communication + C_verification + 
                            C_integration + C_risk + C_latency_penalty
        """
        components = CostComponents()
        
        # Dynamic Pricing (Eq. 7): Supply-demand based pricing
        components.base_price = self._calculate_dynamic_pricing(contractor, task)
        
        # Communication Cost (Eq. 8): Data transfer and protocol overhead
        components.communication_cost = self._calculate_communication_cost(task, client, contractor)
        
        # Verification Cost: Quality assurance overhead
        components.verification_cost = self._calculate_verification_cost(contractor, task)
        
        # Integration Cost: API and result processing overhead
        components.integration_cost = INTEGRATION_BASE_COST
        
        # Risk Cost (Eq. 9): Multi-factor risk assessment
        components.risk_cost = self._calculate_risk_cost(contractor, task)
        
        # Latency Penalty: Cost of exceeding target completion time
        components.latency_penalty = self._calculate_latency_penalty(contractor, task)
        
        return components
    
    def _estimate_flops(self, task: Task) -> float:
        """Estimate FLOPS requirement based on task characteristics."""
        base_flops = {
            'financial_rag': 2.8e15,
            'risk_assessment': 1.5e15,
            'sentiment_analysis': 8.2e14,
            'portfolio_optimization': 3.1e15
        }
        
        task_flops = base_flops.get(task.task_type, 1.0e15)
        
        # Scale by data size and complexity
        size_multiplier = task.data_size_gb / 8.2  # Normalize to paper's 8.2GB baseline
        complexity_multiplier = task.complexity_factor
        
        return task_flops * size_multiplier * complexity_multiplier
    
    def _calculate_memory_usage(self, task: Task, agent: ClientAgent) -> float:
        """Calculate memory requirements in GB."""
        # Model weights (FinBERT-Large baseline: 1.3GB)
        model_memory = 1.3 * task.model_size_factor
        
        # KV cache scales with document count
        kv_cache_memory = (task.document_count / 50000) * 4.2  # 4.2GB for 50k docs
        
        # Activation memory
        activation_memory = 2.8 * task.complexity_factor
        
        return model_memory + kv_cache_memory + activation_memory
    
    def _calculate_opportunity_cost(self, agent: ClientAgent, execution_time_hours: float) -> float:
        """Calculate opportunity cost based on pending high-value tasks."""
        if not agent.task_queue:
            return 0.0
        
        # Find maximum value rate from pending tasks
        max_value_rate = max(
            task.value / task.estimated_duration_hours 
            for task in agent.task_queue
        )
        
        return max_value_rate * execution_time_hours
    
    def _calculate_depreciation_cost(self, agent: ClientAgent, execution_time_hours: float) -> float:
        """Calculate usage-based hardware depreciation."""
        hardware_value = agent.hardware.purchase_cost
        depreciation_period_hours = 3 * 365 * 24  # 3 years
        hourly_depreciation = hardware_value / depreciation_period_hours
        
        return hourly_depreciation * execution_time_hours
    
    def _calculate_dynamic_pricing(self, contractor: ContractorAgent, task: Task) -> float:
        """
        Calculate dynamic pricing based on Eq. (7) from the paper.
        
        P_j(T) = P_base × (1 + α × (D_current - S_available)/S_total) × β_complexity(T)
        """
        base_price = contractor.base_price_per_task
        
        # Supply-demand adjustment
        demand_factor = contractor.current_demand
        supply_factor = contractor.available_capacity / contractor.total_capacity
        alpha = 0.5  # Demand sensitivity factor
        
        demand_multiplier = 1 + alpha * (demand_factor - supply_factor)
        
        # Task complexity multiplier
        complexity_multiplier = 1 + (task.complexity_factor - 1) * 0.3
        
        return base_price * demand_multiplier * complexity_multiplier
    
    def _calculate_communication_cost(self, task: Task, client: ClientAgent, 
                                    contractor: ContractorAgent) -> float:
        """Calculate data transfer and protocol overhead costs."""
        # Data transfer cost
        total_data_gb = task.data_size_gb + task.expected_output_size_gb
        bandwidth_mbps = min(client.network.bandwidth_mbps, contractor.network.bandwidth_mbps)
        transfer_time_hours = (total_data_gb * 8 * 1024) / (bandwidth_mbps * 3600)  # Convert to hours
        
        transfer_cost = total_data_gb * DATA_TRANSFER_COST_PER_GB
        
        # Protocol overhead (encryption, authentication, etc.)
        protocol_overhead = PROTOCOL_OVERHEAD_BASE_COST
        
        return transfer_cost + protocol_overhead
    
    def _calculate_verification_cost(self, contractor: ContractorAgent, task: Task) -> float:
        """Calculate quality verification overhead."""
        if contractor.reliability_score > 0.95:
            return VERIFICATION_AUTOMATED_COST  # High-reliability agents use automated verification
        else:
            return VERIFICATION_MANUAL_COST  # Lower-reliability agents require manual verification
    
    def _calculate_risk_cost(self, contractor: ContractorAgent, task: Task) -> float:
        """
        Calculate risk cost using Eq. (9) from the paper.
        
        C_risk = V_task × (P_failure + P_security + P_quality) × γ_impact
        """
        task_value = task.value
        
        # Risk probabilities
        failure_prob = 1 - contractor.reliability_score
        security_prob = contractor.security_risk_score
        quality_prob = contractor.quality_risk_score
        
        total_risk_prob = failure_prob + security_prob + quality_prob
        
        # Impact severity multiplier
        impact_multiplier = 1.5 if task.criticality == 'high' else 1.0
        
        return task_value * total_risk_prob * impact_multiplier
    
    def _calculate_latency_penalty(self, contractor: ContractorAgent, task: Task) -> float:
        """Calculate penalty for exceeding target latency."""
        estimated_latency_minutes = contractor.avg_latency_minutes
        target_latency_minutes = task.max_latency_minutes
        
        if estimated_latency_minutes > target_latency_minutes:
            excess_minutes = estimated_latency_minutes - target_latency_minutes
            return excess_minutes * LATENCY_PENALTY_PER_MINUTE
        
        return 0.0
    
    def calibrate_costs(self, agent_id: str, estimated_cost: float, actual_cost: float):
        """
        Calibrate cost estimates using EWMA as per Eq. (10) from the paper.
        
        Ĉ_t = λ × C_actual,t-1 + (1-λ) × Ĉ_t-1
        """
        if agent_id not in self.calibration_history:
            self.calibration_history[agent_id] = []
        
        history = self.calibration_history[agent_id]
        
        if history:
            # Apply EWMA calibration
            previous_estimate = history[-1]
            calibrated_estimate = (self.ewma_lambda * actual_cost + 
                                 (1 - self.ewma_lambda) * previous_estimate)
        else:
            calibrated_estimate = actual_cost
        
        history.append(calibrated_estimate)
        
        # Keep only recent history (last 100 entries)
        if len(history) > 100:
            history.pop(0)
        
        return calibrated_estimate