"""
Real Agent Marketplace Implementation

This module implements the marketplace where real LLM agents can:
- Discover each other
- Negotiate task outsourcing
- Track performance and reputation
- Handle payments and transactions
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import uuid

from .base_agent import RealAgent, RealTask, TaskResult, TaskProposal


@dataclass
class MarketplaceTransaction:
    """Record of a transaction in the marketplace."""
    transaction_id: str
    client_agent_id: str
    contractor_agent_id: str
    task_id: str
    amount: float
    timestamp: datetime
    status: str  # 'pending', 'completed', 'failed'


@dataclass
class MarketplaceStats:
    """Statistics about marketplace activity."""
    total_agents: int
    active_agents: int
    total_tasks: int
    successful_tasks: int
    total_volume: float
    avg_task_cost: float
    avg_outsourcing_rate: float


class RealAgentMarketplace:
    """
    The marketplace where real LLM agents discover each other and trade tasks.
    
    This is the breakthrough component - a functioning marketplace where
    autonomous agents make real economic decisions using COALESCE.
    """
    
    def __init__(self, marketplace_id: str = None):
        self.marketplace_id = marketplace_id or str(uuid.uuid4())
        self.agents: Dict[str, RealAgent] = {}
        self.active_tasks: Dict[str, RealTask] = {}
        self.completed_tasks: List[TaskResult] = []
        self.transactions: List[MarketplaceTransaction] = []
        
        # Marketplace state
        self.is_running = False
        self.start_time = None
        
        # Performance tracking
        self.performance_history = []
        
        # Logging
        self.logger = logging.getLogger(f"Marketplace.{self.marketplace_id[:8]}")
        
    async def start(self):
        """Start the marketplace."""
        self.is_running = True
        self.start_time = datetime.now()
        self.logger.info(f"Marketplace {self.marketplace_id[:8]} started")
        
        # Start background tasks
        asyncio.create_task(self._performance_monitor())
        
    async def stop(self):
        """Stop the marketplace."""
        self.is_running = False
        self.logger.info(f"Marketplace {self.marketplace_id[:8]} stopped")
        
    async def register_agent(self, agent: RealAgent):
        """Register an agent with the marketplace."""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Agent {agent.name} registered (total: {len(self.agents)})")
        
    async def unregister_agent(self, agent_id: str):
        """Remove an agent from the marketplace."""
        if agent_id in self.agents:
            agent_name = self.agents[agent_id].name
            del self.agents[agent_id]
            self.logger.info(f"Agent {agent_name} unregistered")
            
    async def get_available_agents(self, exclude: str = None) -> List[RealAgent]:
        """Get list of available agents, optionally excluding one."""
        agents = []
        for agent_id, agent in self.agents.items():
            if exclude and agent_id == exclude:
                continue
            agents.append(agent)
        return agents
        
    async def submit_task_to_marketplace(self, task: RealTask, client_agent: RealAgent) -> TaskResult:
        """
        Submit a task to the marketplace for execution.
        
        This is where the COALESCE magic happens - real agents making real decisions
        about task outsourcing in a functioning marketplace.
        """
        self.logger.info(f"Task {task.task_id} submitted by {client_agent.name}")
        
        # Add to active tasks
        self.active_tasks[task.task_id] = task
        
        try:
            # Let the client agent handle the COALESCE decision process
            result = await client_agent.submit_task(task)
            
            # Record the completed task
            self.completed_tasks.append(result)
            
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
                
            # Update marketplace statistics
            await self._update_marketplace_stats(task, result, client_agent)
            
            self.logger.info(f"Task {task.task_id} completed: {result.success}")
            return result
            
        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")
            # Clean up
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            raise
            
    async def get_marketplace_stats(self) -> MarketplaceStats:
        """Get current marketplace statistics."""
        total_agents = len(self.agents)
        active_agents = len([a for a in self.agents.values() if len(a.active_tasks) > 0])
        total_tasks = len(self.completed_tasks)
        successful_tasks = len([t for t in self.completed_tasks if t.success])
        
        total_volume = sum(t.actual_cost for t in self.completed_tasks if t.success)
        avg_task_cost = total_volume / max(successful_tasks, 1)
        
        # Calculate average outsourcing rate
        outsourcing_rates = []
        for agent in self.agents.values():
            perf = agent.get_performance_summary()
            if perf['total_tasks'] > 0:
                outsourcing_rates.append(perf['outsourcing_rate'])
                
        avg_outsourcing_rate = sum(outsourcing_rates) / max(len(outsourcing_rates), 1)
        
        return MarketplaceStats(
            total_agents=total_agents,
            active_agents=active_agents,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            total_volume=total_volume,
            avg_task_cost=avg_task_cost,
            avg_outsourcing_rate=avg_outsourcing_rate
        )
        
    async def get_agent_leaderboard(self) -> List[Dict]:
        """Get agent performance leaderboard."""
        leaderboard = []
        
        for agent in self.agents.values():
            perf = agent.get_performance_summary()
            if perf['total_tasks'] > 0:
                leaderboard.append({
                    'agent_name': agent.name,
                    'agent_id': agent.agent_id,
                    'total_tasks': perf['total_tasks'],
                    'success_rate': perf['success_rate'],
                    'avg_quality': perf['avg_quality_score'],
                    'reputation': perf['reputation_score'],
                    'net_profit': perf['net_profit'],
                    'outsourcing_rate': perf['outsourcing_rate']
                })
                
        # Sort by reputation score
        leaderboard.sort(key=lambda x: x['reputation'], reverse=True)
        return leaderboard
        
    async def run_marketplace_experiment(self, duration_minutes: int = 30, 
                                       task_generation_rate: float = 2.0) -> Dict:
        """
        Run a live marketplace experiment with real agents.
        
        This generates the breakthrough results - real agents using COALESCE
        to make real decisions in a functioning marketplace.
        """
        self.logger.info(f"Starting {duration_minutes}-minute marketplace experiment")
        
        if not self.is_running:
            await self.start()
            
        experiment_start = datetime.now()
        experiment_end = experiment_start + timedelta(minutes=duration_minutes)
        
        # Track experiment results
        experiment_results = {
            'start_time': experiment_start,
            'duration_minutes': duration_minutes,
            'tasks_generated': 0,
            'tasks_completed': 0,
            'total_cost_savings': 0.0,
            'total_time_savings': 0.0,
            'agent_performance': {},
            'marketplace_stats': []
        }
        
        # Generate tasks at specified rate
        task_generation_interval = 60.0 / task_generation_rate  # seconds between tasks
        
        async def generate_tasks():
            """Generate tasks for the experiment."""
            task_count = 0
            while datetime.now() < experiment_end:
                # Generate a random task
                task = await self._generate_random_task(task_count)
                
                # Assign to a random agent
                if self.agents:
                    client_agent = list(self.agents.values())[task_count % len(self.agents)]
                    
                    try:
                        # Submit task to marketplace
                        result = await self.submit_task_to_marketplace(task, client_agent)
                        experiment_results['tasks_completed'] += 1
                        
                        # Track cost and time savings
                        if hasattr(result, 'cost_savings'):
                            experiment_results['total_cost_savings'] += result.cost_savings
                        if hasattr(result, 'time_savings'):
                            experiment_results['total_time_savings'] += result.time_savings
                            
                    except Exception as e:
                        self.logger.error(f"Task {task.task_id} failed: {e}")
                        
                task_count += 1
                experiment_results['tasks_generated'] = task_count
                
                # Wait before generating next task
                await asyncio.sleep(task_generation_interval)
                
        # Start task generation
        task_generator = asyncio.create_task(generate_tasks())
        
        # Monitor marketplace stats during experiment
        async def monitor_stats():
            """Monitor marketplace statistics during experiment."""
            while datetime.now() < experiment_end:
                stats = await self.get_marketplace_stats()
                experiment_results['marketplace_stats'].append({
                    'timestamp': datetime.now(),
                    'stats': asdict(stats)
                })
                await asyncio.sleep(30)  # Update every 30 seconds
                
        stats_monitor = asyncio.create_task(monitor_stats())
        
        # Wait for experiment to complete
        await asyncio.sleep(duration_minutes * 60)
        
        # Cancel background tasks
        task_generator.cancel()
        stats_monitor.cancel()
        
        # Collect final results
        experiment_results['end_time'] = datetime.now()
        experiment_results['final_marketplace_stats'] = asdict(await self.get_marketplace_stats())
        experiment_results['agent_leaderboard'] = await self.get_agent_leaderboard()
        
        # Calculate aggregate performance metrics
        if experiment_results['tasks_completed'] > 0:
            experiment_results['avg_cost_reduction'] = (
                experiment_results['total_cost_savings'] / experiment_results['tasks_completed']
            ) * 100
            experiment_results['avg_time_savings'] = (
                experiment_results['total_time_savings'] / experiment_results['tasks_completed']
            ) * 100
        else:
            experiment_results['avg_cost_reduction'] = 0.0
            experiment_results['avg_time_savings'] = 0.0
            
        self.logger.info(f"Experiment completed: {experiment_results['tasks_completed']} tasks")
        return experiment_results
        
    async def _generate_random_task(self, task_id: int) -> RealTask:
        """Generate a random task for testing."""
        task_types = ['document_analysis', 'document_summarization', 'code_generation', 
                     'test_generation', 'data_processing', 'data_analysis']
        
        task_type = task_types[task_id % len(task_types)]
        
        if task_type == 'document_analysis':
            input_data = {
                'text': f"This is a sample document for analysis task {task_id}. " * 50
            }
            requirements = {'analysis_type': 'general', 'complexity': 1.0}
            
        elif task_type == 'document_summarization':
            input_data = {
                'text': f"This is a longer document that needs summarization for task {task_id}. " * 100
            }
            requirements = {'summary_length': 'medium', 'complexity': 0.8}
            
        elif task_type == 'code_generation':
            input_data = {
                'requirements': f"Create a function that processes data for task {task_id}"
            }
            requirements = {'language': 'python', 'style': 'clean', 'complexity': 1.2}
            
        elif task_type == 'data_processing':
            input_data = {
                'data_source': {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
            }
            requirements = {'operations': ['clean_nulls'], 'complexity': 0.9}
            
        else:
            input_data = {'sample': f'data for task {task_id}'}
            requirements = {'complexity': 1.0}
            
        return RealTask(
            task_id=f"task_{task_id:04d}",
            task_type=task_type,
            description=f"Generated task {task_id} of type {task_type}",
            input_data=input_data,
            requirements=requirements,
            max_cost=1.0,
            deadline=datetime.now() + timedelta(minutes=10)
        )
        
    async def _update_marketplace_stats(self, task: RealTask, result: TaskResult, 
                                      client_agent: RealAgent):
        """Update marketplace statistics after task completion."""
        # Record transaction if task was outsourced
        # This would be more sophisticated in a real implementation
        pass
        
    async def _performance_monitor(self):
        """Background task to monitor marketplace performance."""
        while self.is_running:
            try:
                stats = await self.get_marketplace_stats()
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'stats': asdict(stats)
                })
                
                # Keep only last 1000 records
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
                    
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                
            await asyncio.sleep(60)  # Update every minute
            
    def export_experiment_results(self, results: Dict, filename: str):
        """Export experiment results to JSON file."""
        # Convert datetime objects to strings for JSON serialization
        def datetime_converter(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object {obj} is not JSON serializable")
            
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=datetime_converter)
            
        self.logger.info(f"Experiment results exported to {filename}")