"""
Base Real Agent Implementation for COALESCE

This module defines the base class for real LLM agents that can participate
in the COALESCE marketplace, make actual decisions, and execute real tasks.
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging

# Import COALESCE decision engine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from decision.decision_engine import DecisionEngine, DecisionResult
from cost_model.cost_calculator import CostCalculator


@dataclass
class RealTask:
    """Represents a real task that can be executed by agents."""
    task_id: str
    task_type: str
    description: str
    input_data: Any
    requirements: Dict[str, Any]
    max_cost: float
    deadline: datetime
    quality_threshold: float = 0.8
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class AgentCapability:
    """Describes what an agent can do and at what cost/quality."""
    capability_name: str
    skill_level: float  # 0.0 to 1.0
    cost_per_unit: float
    avg_execution_time: float
    quality_score: float
    max_concurrent_tasks: int


@dataclass
class TaskProposal:
    """A proposal from one agent to execute a task for another."""
    proposal_id: str
    task_id: str
    contractor_id: str
    estimated_cost: float
    estimated_time: float
    quality_guarantee: float
    expires_at: datetime


@dataclass
class TaskResult:
    """Result of executing a real task."""
    task_id: str
    success: bool
    output_data: Any
    actual_cost: float
    actual_time: float
    quality_score: float
    error_message: Optional[str] = None
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now()


class RealAgent(ABC):
    """
    Base class for real LLM agents in the COALESCE marketplace.
    
    This class provides the core functionality for:
    - Task discovery and negotiation
    - COALESCE-based decision making
    - Real task execution
    - Performance tracking and reputation
    """
    
    def __init__(self, agent_id: str, name: str, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = {cap.capability_name: cap for cap in capabilities}
        
        # COALESCE components
        self.decision_engine = DecisionEngine()
        self.cost_calculator = CostCalculator()
        
        # Agent state
        self.active_tasks: Dict[str, RealTask] = {}
        self.completed_tasks: List[TaskResult] = []
        self.reputation_score = 1.0
        self.total_earnings = 0.0
        self.total_spent = 0.0
        
        # Marketplace connections
        self.known_agents: Dict[str, 'RealAgent'] = {}
        self.pending_proposals: Dict[str, TaskProposal] = {}
        
        # Performance tracking
        self.performance_history = []
        
        # Logging
        self.logger = logging.getLogger(f"RealAgent.{self.name}")
        
    async def register_with_marketplace(self, marketplace):
        """Register this agent with the marketplace for discovery."""
        await marketplace.register_agent(self)
        self.logger.info(f"Agent {self.name} registered with marketplace")
        
    async def discover_agents(self, marketplace) -> List['RealAgent']:
        """Discover other agents in the marketplace."""
        agents = await marketplace.get_available_agents(exclude=self.agent_id)
        for agent in agents:
            self.known_agents[agent.agent_id] = agent
        self.logger.info(f"Discovered {len(agents)} agents in marketplace")
        return agents
        
    async def submit_task(self, task: RealTask) -> TaskResult:
        """
        Submit a task for execution using COALESCE decision framework.
        
        This is where the breakthrough happens - real agents making real decisions
        about whether to execute locally or outsource to other agents.
        """
        self.logger.info(f"Processing task {task.task_id}: {task.description}")
        
        # Step 1: Calculate local execution cost
        local_cost = await self._calculate_local_cost(task)
        
        # Step 2: Get proposals from other agents
        proposals = await self._request_proposals(task)
        
        # Step 3: Use COALESCE decision engine to choose
        decision = await self._make_coalesce_decision(task, local_cost, proposals)
        
        # Step 4: Execute based on decision
        if decision.decision == 'LOCAL':
            result = await self._execute_locally(task)
            self.logger.info(f"Executed task {task.task_id} locally")
        else:
            result = await self._outsource_task(task, decision.selected_contractor)
            self.logger.info(f"Outsourced task {task.task_id} to {decision.selected_contractor}")
            
        # Step 5: Update performance tracking
        await self._update_performance(task, result, decision)
        
        return result
        
    async def _calculate_local_cost(self, task: RealTask) -> float:
        """Calculate the cost of executing this task locally."""
        if task.task_type not in self.capabilities:
            return float('inf')  # Can't execute locally
            
        capability = self.capabilities[task.task_type]
        
        # Base cost calculation
        base_cost = capability.cost_per_unit
        
        # Adjust for task complexity
        complexity_factor = task.requirements.get('complexity', 1.0)
        
        # Adjust for current load
        load_factor = len(self.active_tasks) / max(capability.max_concurrent_tasks, 1)
        
        total_cost = base_cost * complexity_factor * (1 + load_factor)
        
        return total_cost
        
    async def _request_proposals(self, task: RealTask) -> List[TaskProposal]:
        """Request proposals from other agents for this task."""
        proposals = []
        
        for agent_id, agent in self.known_agents.items():
            if await agent._can_execute_task(task):
                proposal = await agent._generate_proposal(task)
                if proposal:
                    proposals.append(proposal)
                    
        self.logger.info(f"Received {len(proposals)} proposals for task {task.task_id}")
        return proposals
        
    async def _make_coalesce_decision(self, task: RealTask, local_cost: float, 
                                    proposals: List[TaskProposal]) -> DecisionResult:
        """
        Use the COALESCE decision engine to choose between local execution
        and outsourcing to other agents.
        
        This is the core breakthrough - real agents using COALESCE for real decisions.
        """
        # Convert proposals to contractor candidates
        candidates = []
        for proposal in proposals:
            contractor = self.known_agents.get(proposal.contractor_id)
            if contractor:
                candidates.append(contractor)
                
        # Create a mock client agent for the decision engine
        # (In real implementation, this would be more sophisticated)
        mock_client = type('MockClient', (), {
            'agent_id': self.agent_id,
            'current_load': len(self.active_tasks)
        })()
        
        # Convert RealTask to simulation Task format
        mock_task = type('MockTask', (), {
            'task_type': task.task_type,
            'complexity_factor': task.requirements.get('complexity', 1.0),
            'value': task.max_cost,
            'data_size_gb': task.requirements.get('data_size_gb', 1.0)
        })()
        
        # Use COALESCE decision engine
        decision = self.decision_engine.make_decision(mock_client, mock_task, candidates)
        
        self.logger.info(f"COALESCE decision for task {task.task_id}: {decision.decision}")
        return decision
        
    async def _execute_locally(self, task: RealTask) -> TaskResult:
        """Execute the task locally using this agent's capabilities."""
        start_time = time.time()
        
        try:
            # This is where the real LLM execution would happen
            output = await self._perform_real_execution(task)
            
            execution_time = time.time() - start_time
            actual_cost = await self._calculate_actual_cost(task, execution_time)
            quality_score = await self._evaluate_quality(task, output)
            
            result = TaskResult(
                task_id=task.task_id,
                success=True,
                output_data=output,
                actual_cost=actual_cost,
                actual_time=execution_time,
                quality_score=quality_score
            )
            
            self.completed_tasks.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Local execution failed for task {task.task_id}: {e}")
            return TaskResult(
                task_id=task.task_id,
                success=False,
                output_data=None,
                actual_cost=0.0,
                actual_time=time.time() - start_time,
                quality_score=0.0,
                error_message=str(e)
            )
            
    async def _outsource_task(self, task: RealTask, contractor_agent) -> TaskResult:
        """Outsource the task to another agent."""
        start_time = time.time()
        
        try:
            # Send task to contractor
            result = await contractor_agent._execute_as_contractor(task, self)
            
            # Pay the contractor
            if result.success:
                await self._make_payment(contractor_agent, result.actual_cost)
                self.total_spent += result.actual_cost
                
            return result
            
        except Exception as e:
            self.logger.error(f"Outsourcing failed for task {task.task_id}: {e}")
            return TaskResult(
                task_id=task.task_id,
                success=False,
                output_data=None,
                actual_cost=0.0,
                actual_time=time.time() - start_time,
                quality_score=0.0,
                error_message=str(e)
            )
            
    async def _execute_as_contractor(self, task: RealTask, client_agent) -> TaskResult:
        """Execute a task as a contractor for another agent."""
        self.logger.info(f"Executing task {task.task_id} as contractor for {client_agent.name}")
        
        # Execute the task
        result = await self._execute_locally(task)
        
        # Update earnings
        if result.success:
            self.total_earnings += result.actual_cost
            
        return result
        
    @abstractmethod
    async def _perform_real_execution(self, task: RealTask) -> Any:
        """
        Perform the actual task execution using real LLM APIs.
        This must be implemented by concrete agent classes.
        """
        pass
        
    async def _can_execute_task(self, task: RealTask) -> bool:
        """Check if this agent can execute the given task."""
        return task.task_type in self.capabilities
        
    async def _generate_proposal(self, task: RealTask) -> Optional[TaskProposal]:
        """Generate a proposal to execute a task for another agent."""
        if not await self._can_execute_task(task):
            return None
            
        capability = self.capabilities[task.task_type]
        
        # Calculate proposal cost
        estimated_cost = await self._calculate_local_cost(task)
        
        # Add profit margin
        estimated_cost *= 1.2  # 20% profit margin
        
        # Check if we can meet the deadline
        estimated_time = capability.avg_execution_time
        
        proposal = TaskProposal(
            proposal_id=str(uuid.uuid4()),
            task_id=task.task_id,
            contractor_id=self.agent_id,
            estimated_cost=estimated_cost,
            estimated_time=estimated_time,
            quality_guarantee=capability.quality_score,
            expires_at=datetime.now()
        )
        
        return proposal
        
    async def _calculate_actual_cost(self, task: RealTask, execution_time: float) -> float:
        """Calculate the actual cost of executing a task."""
        capability = self.capabilities[task.task_type]
        return capability.cost_per_unit * execution_time
        
    async def _evaluate_quality(self, task: RealTask, output: Any) -> float:
        """Evaluate the quality of task execution output."""
        # This would implement real quality evaluation
        # For now, return a simulated quality score
        return 0.85 + (0.15 * self.reputation_score)
        
    async def _make_payment(self, contractor_agent, amount: float):
        """Make payment to a contractor agent."""
        # In a real implementation, this would handle actual payments
        self.logger.info(f"Paid {amount} to {contractor_agent.name}")
        
    async def _update_performance(self, task: RealTask, result: TaskResult, decision: DecisionResult):
        """Update performance tracking and reputation."""
        performance_record = {
            'timestamp': datetime.now(),
            'task_id': task.task_id,
            'decision': decision.decision,
            'success': result.success,
            'cost_savings': decision.cost_savings if hasattr(decision, 'cost_savings') else 0.0,
            'quality_score': result.quality_score
        }
        
        self.performance_history.append(performance_record)
        
        # Update reputation based on success rate
        recent_tasks = self.performance_history[-10:]  # Last 10 tasks
        success_rate = sum(1 for p in recent_tasks if p['success']) / len(recent_tasks)
        self.reputation_score = 0.5 + (0.5 * success_rate)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of this agent's performance."""
        if not self.performance_history:
            return {'total_tasks': 0}
            
        total_tasks = len(self.performance_history)
        successful_tasks = sum(1 for p in self.performance_history if p['success'])
        avg_quality = sum(p['quality_score'] for p in self.performance_history) / total_tasks
        
        outsourced_tasks = sum(1 for p in self.performance_history if p['decision'] == 'OUTSOURCE')
        outsourcing_rate = outsourced_tasks / total_tasks
        
        return {
            'total_tasks': total_tasks,
            'success_rate': successful_tasks / total_tasks,
            'avg_quality_score': avg_quality,
            'outsourcing_rate': outsourcing_rate,
            'reputation_score': self.reputation_score,
            'total_earnings': self.total_earnings,
            'total_spent': self.total_spent,
            'net_profit': self.total_earnings - self.total_spent
        }