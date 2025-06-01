"""
COALESCE Decision Engine implementing the advanced multi-criteria decision algorithm.

This module implements the TOPSIS-based decision framework from the paper,
including skill compatibility assessment, dynamic weight calculation, and
game-theoretic optimization.
"""

import numpy as np
import logging
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

from ..agents.agent_types import ClientAgent, ContractorAgent, Task
from ..cost_model.cost_calculator import CostCalculator, CostComponents
from ..utils.constants import *


@dataclass
class DecisionCriteria:
    """Multi-criteria decision analysis criteria."""
    cost: float
    reliability: float
    latency: float
    security: float
    skill_compatibility: float


@dataclass
class DecisionResult:
    """Result of the decision-making process."""
    decision: str  # 'LOCAL' or 'OUTSOURCE'
    selected_contractor: Optional[ContractorAgent] = None
    topsis_score: float = 0.0
    confidence: float = 0.0
    cost_savings: float = 0.0
    time_savings: float = 0.0
    criteria_scores: Optional[DecisionCriteria] = None
    exploration: bool = False  # Whether this was an exploration decision


class DecisionEngine:
    """
    Advanced multi-criteria decision engine implementing the COALESCE algorithm.
    
    Based on Algorithm 1 from Section 3.6 of the paper.
    """
    
    def __init__(self):
        self.cost_calculator = CostCalculator()
        self.logger = logging.getLogger(__name__)
        self.decision_history: List[DecisionResult] = []
        
        # TOPSIS parameters
        self.skill_threshold = 0.7  # θ_skill from the algorithm
        self.topsis_threshold = 0.6  # τ_threshold from the algorithm
        self.min_confidence = 0.8   # ρ_min from the algorithm
        
        # Epsilon-greedy exploration parameters
        self.exploration_rate = 0.1  # 10% exploration rate
        
        # Dynamic weight learning parameters
        self.learning_rate = 0.01   # η from Eq. (11)
        self.weights = np.array([0.4, 0.25, 0.2, 0.15])  # [cost, reliability, latency, security]
    
    def make_decision(self, client: ClientAgent, task: Task, 
                     candidates: List[ContractorAgent]) -> DecisionResult:
        """
        Execute the COALESCE decision algorithm.
        
        Implements Algorithm 1 from the paper with all phases:
        1. Multi-dimensional Cost Analysis
        2. Dynamic Weight Calculation  
        3. TOPSIS Multi-Criteria Analysis
        4. Game-Theoretic Optimization
        5. Final Decision with Confidence Interval
        """
        self.logger.debug(f"Making decision for task {task.task_id} with {len(candidates)} candidates")
        
        # Phase 1: Multi-dimensional Cost Analysis
        internal_cost = self.cost_calculator.calculate_internal_cost(client, task)
        
        # Filter eligible candidates based on skill compatibility
        eligible_candidates = []
        criteria_matrix = []
        
        for contractor in candidates:
            skill_compatibility = self._assess_skill_compatibility(contractor, task)
            
            if skill_compatibility >= self.skill_threshold:
                external_cost = self.cost_calculator.calculate_external_cost(contractor, task, client)
                reliability = contractor.reliability_score
                latency = contractor.avg_latency_minutes
                security = 1.0 - contractor.security_risk_score  # Convert risk to security score
                
                eligible_candidates.append(contractor)
                criteria_matrix.append([
                    external_cost.external_total,
                    reliability,
                    latency,
                    security
                ])
        
        # Epsilon-greedy exploration: Try random contractor even if no eligible candidates
        if random.random() < self.exploration_rate and candidates:
            # EXPLORATION: Select a random contractor to learn from
            exploration_contractor = random.choice(candidates)
            external_cost = self.cost_calculator.calculate_external_cost(exploration_contractor, task, client)
            
            cost_savings = internal_cost.internal_total - external_cost.external_total
            time_savings = self._calculate_time_savings(client, exploration_contractor, task)
            
            self.logger.debug(f"Exploration decision: trying contractor {exploration_contractor.name}")
            
            return DecisionResult(
                decision='OUTSOURCE',
                selected_contractor=exploration_contractor,
                topsis_score=0.5,  # Moderate exploration score
                confidence=0.7,    # Lower confidence for exploration
                cost_savings=cost_savings,
                time_savings=time_savings,
                exploration=True,
                criteria_scores=DecisionCriteria(
                    cost=external_cost.external_total,
                    reliability=exploration_contractor.reliability_score,
                    latency=exploration_contractor.avg_latency_minutes,
                    security=1.0 - exploration_contractor.security_risk_score,
                    skill_compatibility=self._assess_skill_compatibility(exploration_contractor, task)
                )
            )
        
        if not eligible_candidates:
            return DecisionResult(
                decision='LOCAL',
                confidence=1.0,
                criteria_scores=DecisionCriteria(
                    cost=internal_cost.internal_total,
                    reliability=0.95,  # Assume high local reliability
                    latency=self._estimate_local_latency(task),
                    security=0.98,     # Assume high local security
                    skill_compatibility=1.0
                )
            )
        
        # Phase 2: Dynamic Weight Calculation
        weights = self._calculate_dynamic_weights(task, client)
        
        # Phase 3: TOPSIS Multi-Criteria Analysis
        topsis_scores = self._calculate_topsis_scores(criteria_matrix, weights)
        
        # Phase 4: Game-Theoretic Optimization
        best_idx = np.argmax(topsis_scores)
        best_contractor = eligible_candidates[best_idx]
        best_score = topsis_scores[best_idx]
        
        nash_strategy = self._calculate_nash_strategy(
            internal_cost.internal_total,
            criteria_matrix[best_idx][0]  # External cost
        )
        
        # Phase 5: Final Decision with Confidence Interval
        confidence = self._calculate_decision_confidence(task, best_contractor)
        
        # Make final decision
        if best_score > self.topsis_threshold and confidence > self.min_confidence:
            external_cost_components = self.cost_calculator.calculate_external_cost(
                best_contractor, task, client
            )
            
            cost_savings = internal_cost.internal_total - external_cost_components.external_total
            time_savings = self._calculate_time_savings(client, best_contractor, task)
            
            result = DecisionResult(
                decision='OUTSOURCE',
                selected_contractor=best_contractor,
                topsis_score=best_score,
                confidence=confidence,
                cost_savings=cost_savings,
                time_savings=time_savings,
                criteria_scores=DecisionCriteria(
                    cost=external_cost_components.external_total,
                    reliability=best_contractor.reliability_score,
                    latency=best_contractor.avg_latency_minutes,
                    security=1.0 - best_contractor.security_risk_score,
                    skill_compatibility=self._assess_skill_compatibility(best_contractor, task)
                )
            )
        else:
            result = DecisionResult(
                decision='LOCAL',
                confidence=confidence,
                criteria_scores=DecisionCriteria(
                    cost=internal_cost.internal_total,
                    reliability=0.95,
                    latency=self._estimate_local_latency(task),
                    security=0.98,
                    skill_compatibility=1.0
                )
            )
        
        # Store decision for learning
        self.decision_history.append(result)
        self._update_weights(result, task)
        
        return result
    
    def _assess_skill_compatibility(self, contractor: ContractorAgent, task: Task) -> float:
        """
        Assess skill compatibility using Eq. (11) from the paper.
        
        Skill_Compatibility(A_j, T) = α × S_ontological + β × S_embedding + γ × S_performance
        """
        # Ontological skill match (Jaccard similarity)
        required_skills = set(task.required_skills)
        contractor_skills = set(contractor.skills)
        
        if not required_skills:
            ontological_score = 1.0
        else:
            intersection = len(required_skills.intersection(contractor_skills))
            union = len(required_skills.union(contractor_skills))
            ontological_score = intersection / union if union > 0 else 0.0
        
        # Embedding similarity (cosine similarity simulation)
        embedding_score = self._calculate_embedding_similarity(contractor, task)
        
        # Historical performance score
        performance_score = contractor.performance_scores.get(task.task_type, 0.8)
        
        # Weighted combination (α=0.3, β=0.5, γ=0.2 from paper)
        compatibility = (0.3 * ontological_score + 
                        0.5 * embedding_score + 
                        0.2 * performance_score)
        
        return min(1.0, max(0.0, compatibility))
    
    def _calculate_embedding_similarity(self, contractor: ContractorAgent, task: Task) -> float:
        """Simulate embedding-based similarity calculation."""
        # Simulate cosine similarity based on task type and contractor specialization
        specialization_match = {
            'financial_rag': {
                'gpu_specialist': 0.95,
                'cpu_specialist': 0.7,
                'budget_provider': 0.6,
                'edge_specialist': 0.4,
                'hybrid_cloud': 0.9,
                'quantum_enhanced': 0.8
            },
            'risk_assessment': {
                'gpu_specialist': 0.8,
                'cpu_specialist': 0.9,
                'budget_provider': 0.7,
                'edge_specialist': 0.5,
                'hybrid_cloud': 0.85,
                'quantum_enhanced': 0.95
            }
        }
        
        return specialization_match.get(task.task_type, {}).get(
            contractor.specialization, 0.6
        )
    
    def _calculate_dynamic_weights(self, task: Task, client: ClientAgent) -> np.ndarray:
        """
        Calculate dynamic weights using reinforcement learning approach from Eq. (12).
        
        w_i^(t) = w_i^(t-1) + η × ∇_w Q(s_t, a_t, w_i^(t-1))
        """
        # Adjust weights based on task characteristics and market conditions
        base_weights = self.weights.copy()
        
        # Increase cost weight for budget-sensitive tasks
        if task.budget_constraint < task.value * 0.3:
            base_weights[0] *= 1.2  # Increase cost importance
        
        # Increase latency weight for time-critical tasks
        if task.max_latency_minutes < 30:
            base_weights[2] *= 1.3  # Increase latency importance
        
        # Increase security weight for sensitive tasks
        if task.data_sensitivity == 'high':
            base_weights[3] *= 1.4  # Increase security importance
        
        # Normalize weights
        return base_weights / np.sum(base_weights)
    
    def _calculate_topsis_scores(self, criteria_matrix: List[List[float]], 
                                weights: np.ndarray) -> np.ndarray:
        """
        Calculate TOPSIS scores for multi-criteria decision analysis.
        
        Implements the TOPSIS algorithm from Phase 3 of Algorithm 1.
        """
        if not criteria_matrix:
            return np.array([])
        
        # Convert to numpy array
        matrix = np.array(criteria_matrix)
        
        # Normalize the decision matrix
        scaler = MinMaxScaler()
        normalized_matrix = scaler.fit_transform(matrix)
        
        # Apply weights
        weighted_matrix = normalized_matrix * weights
        
        # Determine ideal and anti-ideal solutions
        # For cost and latency: lower is better (min)
        # For reliability and security: higher is better (max)
        ideal_solution = np.array([
            np.min(weighted_matrix[:, 0]),  # Min cost
            np.max(weighted_matrix[:, 1]),  # Max reliability
            np.min(weighted_matrix[:, 2]),  # Min latency
            np.max(weighted_matrix[:, 3])   # Max security
        ])
        
        anti_ideal_solution = np.array([
            np.max(weighted_matrix[:, 0]),  # Max cost
            np.min(weighted_matrix[:, 1]),  # Min reliability
            np.max(weighted_matrix[:, 2]),  # Max latency
            np.min(weighted_matrix[:, 3])   # Min security
        ])
        
        # Calculate distances and TOPSIS scores
        topsis_scores = []
        for row in weighted_matrix:
            d_positive = euclidean(row, ideal_solution)
            d_negative = euclidean(row, anti_ideal_solution)
            
            if d_positive + d_negative == 0:
                score = 0.5
            else:
                score = d_negative / (d_positive + d_negative)
            
            topsis_scores.append(score)
        
        return np.array(topsis_scores)
    
    def _calculate_nash_strategy(self, internal_cost: float, external_cost: float) -> float:
        """
        Calculate Nash equilibrium strategy from Eq. (14).
        
        Nash_Strategy = argmin_s_c max_s_a U_c(s_c, s_a) - U_a(s_c, s_a)
        """
        # Simplified Nash equilibrium calculation
        # Client utility: negative cost
        # Agent utility: profit margin
        
        client_utility = -external_cost
        agent_utility = external_cost * 0.2  # Assume 20% profit margin
        
        nash_value = client_utility - agent_utility
        return nash_value
    
    def _calculate_decision_confidence(self, task: Task, contractor: ContractorAgent) -> float:
        """
        Calculate decision confidence using bootstrap sampling from Eq. (15).
        
        Confidence = 1 - (2 × min(p, 1-p) / √n) × z_α/2
        """
        if len(self.decision_history) < 10:
            return 0.8  # Default confidence for insufficient history
        
        # Simulate bootstrap sampling of recent decisions
        recent_decisions = self.decision_history[-50:]  # Last 50 decisions
        successful_decisions = sum(1 for d in recent_decisions if d.topsis_score > 0.7)
        
        n = len(recent_decisions)
        p = successful_decisions / n
        
        # Calculate confidence interval (95% confidence, z = 1.96)
        z_alpha_2 = 1.96
        if n > 0:
            confidence = 1 - (2 * min(p, 1-p) / np.sqrt(n)) * z_alpha_2
        else:
            confidence = 0.8  # Default confidence for no data
        
        return min(1.0, max(0.0, confidence))
    
    def _estimate_local_latency(self, task: Task) -> float:
        """Estimate local execution latency in minutes."""
        base_latency = {
            'financial_rag': 125.4,
            'risk_assessment': 90.0,
            'sentiment_analysis': 45.0,
            'portfolio_optimization': 180.0
        }
        
        return base_latency.get(task.task_type, 100.0) * task.complexity_factor
    
    def _calculate_time_savings(self, client: ClientAgent, contractor: ContractorAgent, 
                               task: Task) -> float:
        """Calculate time savings from outsourcing."""
        local_time = self._estimate_local_latency(task)
        contractor_time = contractor.avg_latency_minutes
        
        return max(0.0, local_time - contractor_time)
    
    def _update_weights(self, result: DecisionResult, task: Task):
        """Update weights based on decision outcome using reinforcement learning."""
        if len(self.decision_history) > 1:
            # Simple gradient update based on decision success
            if result.topsis_score > 0.8:
                # Good decision, reinforce current weights
                pass
            else:
                # Poor decision, adjust weights slightly
                self.weights += self.learning_rate * np.random.normal(0, 0.1, len(self.weights))
                self.weights = np.abs(self.weights)  # Ensure positive weights
                weight_sum = np.sum(self.weights)
                if weight_sum > 1e-10:  # Prevent division by zero with small epsilon
                    self.weights /= weight_sum  # Normalize
                else:
                    # Reset to default weights if all become zero
                    self.weights = np.array([0.4, 0.25, 0.2, 0.15])
    
    def get_decision_statistics(self) -> Dict[str, float]:
        """Get statistics about decision-making performance."""
        if not self.decision_history:
            return {}
        
        outsource_decisions = [d for d in self.decision_history if d.decision == 'OUTSOURCE']
        
        return {
            'total_decisions': len(self.decision_history),
            'outsource_rate': len(outsource_decisions) / len(self.decision_history) if self.decision_history else 0.0,
            'avg_topsis_score': np.mean([d.topsis_score for d in outsource_decisions]) if outsource_decisions else 0.0,
            'avg_confidence': np.mean([d.confidence for d in self.decision_history]) if self.decision_history else 0.0,
            'avg_cost_savings': np.mean([d.cost_savings for d in outsource_decisions]) if outsource_decisions else 0.0,
            'avg_time_savings': np.mean([d.time_savings for d in outsource_decisions]) if outsource_decisions else 0.0
        }