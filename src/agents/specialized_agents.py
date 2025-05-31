"""
Specialized agent types for different domains like content creation and customer service.

This module extends the COALESCE framework to support various agent specializations
beyond the financial analysis focus in the original paper.
"""

from dataclasses import dataclass
from typing import Set, Dict
from .agent_types import ContractorAgent, Task, HardwareConfiguration, NetworkConfiguration


def create_content_creation_agents() -> Dict[str, ContractorAgent]:
    """Create specialized content creation contractor agents."""
    
    agents = {}
    
    # Creative Writing Specialist
    agents['creative_writer'] = ContractorAgent(
        name="Creative Writing Specialist",
        specialization="creative_writing",
        base_price_per_task=25.00,
        avg_latency_minutes=45.0,
        reliability_score=0.92,
        market_share=0.08,
        topsis_score=0.78,
        skills={"Creative Writing", "Storytelling", "Content Strategy", "SEO Writing"},
        security_risk_score=0.03,
        quality_risk_score=0.08,
        hardware=HardwareConfiguration(
            gpu_model="NVIDIA RTX 4070",
            gpu_count=1,
            gpu_memory_gb=12,
            gpu_flops=35e12,
            cpu_model="Intel i7-13700K",
            cpu_cores=16,
            cpu_memory_gb=32,
            peak_flops=35e12,
            total_memory_gb=44,
            utilization_factor=0.6,
            tdp_watts=200,
            purchase_cost=3500,
            compute_cost_per_hour=0.80,
            memory_cost_per_hour=0.15
        )
    )
    
    # Video Content Creator
    agents['video_creator'] = ContractorAgent(
        name="Video Content Creator",
        specialization="video_production",
        base_price_per_task=75.00,
        avg_latency_minutes=120.0,
        reliability_score=0.88,
        market_share=0.12,
        topsis_score=0.82,
        skills={"Video Editing", "Motion Graphics", "AI Video Generation", "3D Rendering"},
        security_risk_score=0.05,
        quality_risk_score=0.12,
        hardware=HardwareConfiguration(
            gpu_model="NVIDIA RTX 4090",
            gpu_count=2,
            gpu_memory_gb=48,
            gpu_flops=330e12,
            cpu_model="AMD Ryzen 9 7950X",
            cpu_cores=16,
            cpu_memory_gb=64,
            peak_flops=330e12,
            total_memory_gb=112,
            utilization_factor=0.85,
            tdp_watts=900,
            purchase_cost=8000,
            compute_cost_per_hour=2.20,
            memory_cost_per_hour=0.40
        )
    )
    
    # Social Media Specialist
    agents['social_media'] = ContractorAgent(
        name="Social Media Specialist",
        specialization="social_media",
        base_price_per_task=15.00,
        avg_latency_minutes=20.0,
        reliability_score=0.94,
        market_share=0.15,
        topsis_score=0.85,
        skills={"Social Media Strategy", "Content Scheduling", "Trend Analysis", "Engagement Optimization"},
        security_risk_score=0.02,
        quality_risk_score=0.06,
        hardware=HardwareConfiguration(
            gpu_model="NVIDIA GTX 1660",
            gpu_count=1,
            gpu_memory_gb=6,
            gpu_flops=5e12,
            cpu_model="Intel i5-12400",
            cpu_cores=6,
            cpu_memory_gb=16,
            peak_flops=5e12,
            total_memory_gb=22,
            utilization_factor=0.4,
            tdp_watts=120,
            purchase_cost=1200,
            compute_cost_per_hour=0.35,
            memory_cost_per_hour=0.08
        )
    )
    
    # Set performance scores for content creation tasks
    for agent in agents.values():
        agent.performance_scores = {
            "content_creation": 0.90,
            "video_production": 0.85 if "Video" in agent.skills else 0.60,
            "social_media_management": 0.88 if "Social Media" in agent.skills else 0.65,
            "copywriting": 0.92 if "Writing" in agent.name else 0.70
        }
    
    return agents


def create_customer_service_agents() -> Dict[str, ContractorAgent]:
    """Create specialized customer service contractor agents."""
    
    agents = {}
    
    # Multilingual Support Specialist
    agents['multilingual_support'] = ContractorAgent(
        name="Multilingual Support Specialist",
        specialization="multilingual_support",
        base_price_per_task=30.00,
        avg_latency_minutes=15.0,
        reliability_score=0.96,
        market_share=0.10,
        topsis_score=0.88,
        skills={"Multilingual Support", "Translation", "Cultural Adaptation", "Live Chat"},
        security_risk_score=0.02,
        quality_risk_score=0.04,
        hardware=HardwareConfiguration(
            gpu_model="NVIDIA RTX 3070",
            gpu_count=1,
            gpu_memory_gb=8,
            gpu_flops=20e12,
            cpu_model="Intel i7-12700",
            cpu_cores=12,
            cpu_memory_gb=32,
            peak_flops=20e12,
            total_memory_gb=40,
            utilization_factor=0.5,
            tdp_watts=220,
            purchase_cost=2800,
            compute_cost_per_hour=0.65,
            memory_cost_per_hour=0.12
        )
    )
    
    # Technical Support Specialist
    agents['technical_support'] = ContractorAgent(
        name="Technical Support Specialist",
        specialization="technical_support",
        base_price_per_task=40.00,
        avg_latency_minutes=25.0,
        reliability_score=0.93,
        market_share=0.08,
        topsis_score=0.81,
        skills={"Technical Troubleshooting", "System Diagnostics", "Documentation", "Remote Assistance"},
        security_risk_score=0.03,
        quality_risk_score=0.07,
        hardware=HardwareConfiguration(
            gpu_model="NVIDIA RTX 3060",
            gpu_count=1,
            gpu_memory_gb=12,
            gpu_flops=13e12,
            cpu_model="AMD Ryzen 7 5700X",
            cpu_cores=8,
            cpu_memory_gb=32,
            peak_flops=13e12,
            total_memory_gb=44,
            utilization_factor=0.6,
            tdp_watts=180,
            purchase_cost=2200,
            compute_cost_per_hour=0.55,
            memory_cost_per_hour=0.10
        )
    )
    
    # AI Chatbot Specialist
    agents['ai_chatbot'] = ContractorAgent(
        name="AI Chatbot Specialist",
        specialization="ai_chatbot",
        base_price_per_task=20.00,
        avg_latency_minutes=5.0,
        reliability_score=0.89,
        market_share=0.20,
        topsis_score=0.76,
        skills={"Conversational AI", "Intent Recognition", "Response Generation", "Sentiment Analysis"},
        security_risk_score=0.04,
        quality_risk_score=0.11,
        hardware=HardwareConfiguration(
            gpu_model="NVIDIA T4",
            gpu_count=1,
            gpu_memory_gb=16,
            gpu_flops=8.1e12,
            cpu_model="Intel Xeon E5-2680",
            cpu_cores=14,
            cpu_memory_gb=64,
            peak_flops=8.1e12,
            total_memory_gb=80,
            utilization_factor=0.7,
            tdp_watts=150,
            purchase_cost=4000,
            compute_cost_per_hour=0.45,
            memory_cost_per_hour=0.18
        )
    )
    
    # Set performance scores for customer service tasks
    for agent in agents.values():
        agent.performance_scores = {
            "customer_support": 0.92,
            "technical_troubleshooting": 0.88 if "Technical" in agent.name else 0.70,
            "multilingual_support": 0.95 if "Multilingual" in agent.name else 0.60,
            "chatbot_training": 0.90 if "Chatbot" in agent.name else 0.65,
            "sentiment_analysis": 0.85
        }
    
    return agents


def create_specialized_tasks(task_type: str, **kwargs) -> Task:
    """Create specialized tasks for different domains."""
    
    task_specs = {
        # Content Creation Tasks
        "content_creation": {
            "document_count": 50,
            "data_size_gb": 0.5,
            "complexity_factor": 1.2,
            "value": 120.0,
            "budget_constraint": 80.0,
            "max_latency_minutes": 60.0,
            "required_skills": ["Creative Writing", "Content Strategy", "SEO"]
        },
        "video_production": {
            "document_count": 10,
            "data_size_gb": 15.0,
            "complexity_factor": 2.5,
            "value": 300.0,
            "budget_constraint": 200.0,
            "max_latency_minutes": 180.0,
            "required_skills": ["Video Editing", "Motion Graphics", "3D Rendering"]
        },
        "social_media_management": {
            "document_count": 100,
            "data_size_gb": 1.0,
            "complexity_factor": 0.8,
            "value": 80.0,
            "budget_constraint": 50.0,
            "max_latency_minutes": 30.0,
            "required_skills": ["Social Media Strategy", "Content Scheduling", "Trend Analysis"]
        },
        
        # Customer Service Tasks
        "customer_support": {
            "document_count": 200,
            "data_size_gb": 2.0,
            "complexity_factor": 1.0,
            "value": 100.0,
            "budget_constraint": 60.0,
            "max_latency_minutes": 20.0,
            "required_skills": ["Customer Support", "Communication", "Problem Solving"]
        },
        "technical_troubleshooting": {
            "document_count": 50,
            "data_size_gb": 5.0,
            "complexity_factor": 1.8,
            "value": 180.0,
            "budget_constraint": 120.0,
            "max_latency_minutes": 45.0,
            "required_skills": ["Technical Troubleshooting", "System Diagnostics", "Documentation"]
        },
        "multilingual_support": {
            "document_count": 150,
            "data_size_gb": 3.0,
            "complexity_factor": 1.5,
            "value": 150.0,
            "budget_constraint": 100.0,
            "max_latency_minutes": 25.0,
            "required_skills": ["Multilingual Support", "Translation", "Cultural Adaptation"]
        },
        "chatbot_training": {
            "document_count": 1000,
            "data_size_gb": 8.0,
            "complexity_factor": 2.0,
            "value": 250.0,
            "budget_constraint": 150.0,
            "max_latency_minutes": 90.0,
            "required_skills": ["Conversational AI", "Intent Recognition", "Response Generation"]
        }
    }
    
    spec = task_specs.get(task_type, task_specs["content_creation"])
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if key in spec:
            spec[key] = value
    
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


# Extended task type distributions for different domains
CONTENT_CREATION_TASK_DISTRIBUTION = {
    'content_creation': 0.4,
    'video_production': 0.2,
    'social_media_management': 0.3,
    'copywriting': 0.1
}

CUSTOMER_SERVICE_TASK_DISTRIBUTION = {
    'customer_support': 0.4,
    'technical_troubleshooting': 0.2,
    'multilingual_support': 0.2,
    'chatbot_training': 0.2
}

# Combined distribution for mixed scenarios
MIXED_DOMAIN_TASK_DISTRIBUTION = {
    # Original financial tasks
    'financial_rag': 0.2,
    'risk_assessment': 0.15,
    'sentiment_analysis': 0.1,
    'portfolio_optimization': 0.05,
    
    # Content creation tasks
    'content_creation': 0.2,
    'video_production': 0.1,
    'social_media_management': 0.1,
    
    # Customer service tasks
    'customer_support': 0.15,
    'technical_troubleshooting': 0.1,
    'multilingual_support': 0.1,
    'chatbot_training': 0.05
}