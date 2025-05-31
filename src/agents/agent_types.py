"""
Agent type definitions for the COALESCE simulation.

This module defines the data structures for client agents, contractor agents,
tasks, and hardware configurations as described in the paper.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import uuid
from datetime import datetime


class TaskType(Enum):
    """Task types supported by the COALESCE framework."""
    FINANCIAL_RAG = "financial_rag"
    RISK_ASSESSMENT = "risk_assessment"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"


class AgentSpecialization(Enum):
    """Contractor agent specializations from Table 1 in the paper."""
    GPU_SPECIALIST = "gpu_specialist"
    CPU_SPECIALIST = "cpu_specialist"
    BUDGET_PROVIDER = "budget_provider"
    EDGE_SPECIALIST = "edge_specialist"
    HYBRID_CLOUD = "hybrid_cloud"
    QUANTUM_ENHANCED = "quantum_enhanced"


@dataclass
class HardwareConfiguration:
    """Hardware configuration for agents."""
    # GPU specifications
    gpu_model: str
    gpu_count: int
    gpu_memory_gb: int
    gpu_flops: float  # Peak FLOPS
    
    # CPU specifications
    cpu_model: str
    cpu_cores: int
    cpu_memory_gb: int
    
    # Performance characteristics
    peak_flops: float
    total_memory_gb: float
    utilization_factor: float
    tdp_watts: int
    
    # Cost parameters
    purchase_cost: float
    compute_cost_per_hour: float
    memory_cost_per_hour: float


@dataclass
class NetworkConfiguration:
    """Network configuration for agents."""
    bandwidth_mbps: int
    latency_ms: float
    reliability: float


@dataclass
class Task:
    """Task definition with all parameters needed for cost calculation."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = "financial_rag"
    
    # Data characteristics
    document_count: int = 50000
    data_size_gb: float = 8.2
    expected_output_size_gb: float = 0.2
    
    # Computational requirements
    complexity_factor: float = 1.0
    model_size_factor: float = 1.0
    estimated_duration_hours: float = 2.0
    
    # Business parameters
    value: float = 100.0
    budget_constraint: float = 50.0
    max_latency_minutes: float = 60.0
    criticality: str = "medium"  # low, medium, high
    data_sensitivity: str = "medium"  # low, medium, high
    
    # Requirements
    required_skills: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None


@dataclass
class ClientAgent:
    """Client agent that needs to execute tasks."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Client Agent"
    
    # Hardware and network
    hardware: HardwareConfiguration = None
    network: NetworkConfiguration = None
    
    # Task management
    task_queue: List[Task] = field(default_factory=list)
    completed_tasks: List[Task] = field(default_factory=list)
    
    # Performance tracking
    total_cost_spent: float = 0.0
    total_time_saved: float = 0.0
    outsourcing_rate: float = 0.0
    
    def __post_init__(self):
        if self.hardware is None:
            # Default client hardware (RTX 3080 setup from paper)
            self.hardware = HardwareConfiguration(
                gpu_model="NVIDIA RTX 3080",
                gpu_count=1,
                gpu_memory_gb=10,
                gpu_flops=29.77e12,
                cpu_model="Intel Xeon E5-2690 v4",
                cpu_cores=14,
                cpu_memory_gb=64,
                peak_flops=29.77e12,
                total_memory_gb=74,
                utilization_factor=0.75,
                tdp_watts=320,
                purchase_cost=12000,
                compute_cost_per_hour=0.45,
                memory_cost_per_hour=0.15
            )
        
        if self.network is None:
            self.network = NetworkConfiguration(
                bandwidth_mbps=1000,
                latency_ms=15.0,
                reliability=0.99
            )


@dataclass
class ContractorAgent:
    """Contractor agent that provides services."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Contractor Agent"
    specialization: str = "gpu_specialist"
    
    # Hardware and network
    hardware: HardwareConfiguration = None
    network: NetworkConfiguration = None
    
    # Capabilities
    skills: Set[str] = field(default_factory=set)
    performance_scores: Dict[str, float] = field(default_factory=dict)
    
    # Pricing and availability
    base_price_per_task: float = 20.0
    current_demand: float = 0.5
    available_capacity: float = 0.8
    total_capacity: float = 1.0
    
    # Performance metrics
    reliability_score: float = 0.95
    avg_latency_minutes: float = 30.0
    security_risk_score: float = 0.05
    quality_risk_score: float = 0.03
    
    # Market metrics
    market_share: float = 0.1
    topsis_score: float = 0.8
    
    # Task tracking
    completed_tasks: int = 0
    total_revenue: float = 0.0
    
    def __post_init__(self):
        if self.hardware is None:
            self._set_default_hardware()
        
        if self.network is None:
            self.network = NetworkConfiguration(
                bandwidth_mbps=10000,
                latency_ms=5.0,
                reliability=0.995
            )
    
    def _set_default_hardware(self):
        """Set default hardware based on specialization."""
        hardware_configs = {
            "gpu_specialist": HardwareConfiguration(
                gpu_model="NVIDIA H100",
                gpu_count=8,
                gpu_memory_gb=640,
                gpu_flops=1000e12,
                cpu_model="AMD EPYC 9654",
                cpu_cores=96,
                cpu_memory_gb=1024,
                peak_flops=1000e12,
                total_memory_gb=1664,
                utilization_factor=0.9,
                tdp_watts=2800,
                purchase_cost=200000,
                compute_cost_per_hour=3.50,
                memory_cost_per_hour=0.8
            ),
            "cpu_specialist": HardwareConfiguration(
                gpu_model="None",
                gpu_count=0,
                gpu_memory_gb=0,
                gpu_flops=0,
                cpu_model="Intel Xeon Platinum 8480+",
                cpu_cores=112,
                cpu_memory_gb=512,
                peak_flops=50e12,
                total_memory_gb=512,
                utilization_factor=0.85,
                tdp_watts=1200,
                purchase_cost=80000,
                compute_cost_per_hour=1.20,
                memory_cost_per_hour=0.3
            ),
            "budget_provider": HardwareConfiguration(
                gpu_model="NVIDIA RTX 4090",
                gpu_count=4,
                gpu_memory_gb=96,
                gpu_flops=165e12,
                cpu_model="Intel i9-13900K",
                cpu_cores=24,
                cpu_memory_gb=128,
                peak_flops=165e12,
                total_memory_gb=224,
                utilization_factor=0.7,
                tdp_watts=1600,
                purchase_cost=25000,
                compute_cost_per_hour=1.20,
                memory_cost_per_hour=0.2
            )
        }
        
        self.hardware = hardware_configs.get(self.specialization, hardware_configs["gpu_specialist"])


def create_contractor_from_paper_spec(agent_type: str) -> ContractorAgent:
    """Create contractor agents matching the specifications from Table 1 in the paper."""
    
    specs = {
        "contractor_b": {
            "name": "Premium GPU Specialist",
            "specialization": "gpu_specialist",
            "base_price": 18.50,
            "avg_latency": 8.2,
            "reliability": 0.987,
            "market_share": 15.2,
            "topsis_score": 0.923,
            "skills": {"FinBERT", "Custom RAG", "Financial Analysis", "GPU Computing"}
        },
        "contractor_c": {
            "name": "High-Throughput CPU",
            "specialization": "cpu_specialist", 
            "base_price": 9.60,
            "avg_latency": 52.3,
            "reliability": 0.912,
            "market_share": 22.8,
            "topsis_score": 0.756,
            "skills": {"Parallel Processing", "Ray", "Dask", "CPU Optimization"}
        },
        "contractor_d": {
            "name": "Budget Provider",
            "specialization": "budget_provider",
            "base_price": 6.20,
            "avg_latency": 38.7,
            "reliability": 0.743,
            "market_share": 31.5,
            "topsis_score": 0.612,
            "skills": {"General NLP", "Basic RAG", "Cost Optimization"}
        },
        "contractor_e": {
            "name": "Edge Specialist",
            "specialization": "edge_specialist",
            "base_price": 4.80,
            "avg_latency": 95.1,
            "reliability": 0.856,
            "market_share": 8.7,
            "topsis_score": 0.445,
            "skills": {"Mobile Deployment", "Edge Computing", "Low Latency"}
        },
        "contractor_f": {
            "name": "Hybrid Cloud",
            "specialization": "hybrid_cloud",
            "base_price": 22.40,
            "avg_latency": 12.8,
            "reliability": 0.945,
            "market_share": 18.3,
            "topsis_score": 0.834,
            "skills": {"Auto-scaling", "Multi-region", "Cloud Native", "AWS"}
        },
        "contractor_g": {
            "name": "Quantum-Enhanced",
            "specialization": "quantum_enhanced",
            "base_price": 45.00,
            "avg_latency": 25.6,
            "reliability": 0.892,
            "market_share": 3.5,
            "topsis_score": 0.567,
            "skills": {"Quantum ML", "IBM Quantum", "Research", "Advanced Analytics"}
        }
    }
    
    if agent_type not in specs:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    spec = specs[agent_type]
    
    agent = ContractorAgent(
        name=spec["name"],
        specialization=spec["specialization"],
        base_price_per_task=spec["base_price"],
        avg_latency_minutes=spec["avg_latency"],
        reliability_score=spec["reliability"],
        market_share=spec["market_share"] / 100.0,
        topsis_score=spec["topsis_score"],
        skills=spec["skills"],
        security_risk_score=0.02 + (1 - spec["reliability"]) * 0.1,
        quality_risk_score=0.01 + (1 - spec["reliability"]) * 0.05
    )
    
    # Set performance scores based on specialization
    if "Financial" in spec["skills"] or "FinBERT" in spec["skills"]:
        agent.performance_scores["financial_rag"] = 0.95
        agent.performance_scores["risk_assessment"] = 0.90
    else:
        agent.performance_scores["financial_rag"] = 0.75
        agent.performance_scores["risk_assessment"] = 0.80
    
    agent.performance_scores["sentiment_analysis"] = 0.85
    agent.performance_scores["portfolio_optimization"] = 0.80
    
    return agent


def create_sample_task(task_type: str = "financial_rag") -> Task:
    """Create a sample task matching the paper's specifications."""
    
    task_specs = {
        "financial_rag": {
            "document_count": 50000,
            "data_size_gb": 8.2,
            "complexity_factor": 1.0,
            "value": 150.0,
            "budget_constraint": 75.0,
            "max_latency_minutes": 90.0,
            "required_skills": ["FinBERT", "Financial Analysis", "RAG"]
        },
        "risk_assessment": {
            "document_count": 25000,
            "data_size_gb": 4.1,
            "complexity_factor": 1.2,
            "value": 200.0,
            "budget_constraint": 100.0,
            "max_latency_minutes": 120.0,
            "required_skills": ["Risk Modeling", "Monte Carlo", "Statistics"]
        },
        "sentiment_analysis": {
            "document_count": 100000,
            "data_size_gb": 2.5,
            "complexity_factor": 0.8,
            "value": 80.0,
            "budget_constraint": 40.0,
            "max_latency_minutes": 45.0,
            "required_skills": ["NLP", "Sentiment Analysis", "Text Processing"]
        }
    }
    
    spec = task_specs.get(task_type, task_specs["financial_rag"])
    
    return Task(
        task_type=task_type,
        document_count=spec["document_count"],
        data_size_gb=spec["data_size_gb"],
        complexity_factor=spec["complexity_factor"],
        value=spec["value"],
        budget_constraint=spec["budget_constraint"],
        max_latency_minutes=spec["max_latency_minutes"],
        required_skills=spec["required_skills"]
    )