#!/usr/bin/env python3
"""
Debug skill compatibility calculation to understand why outsourcing rates are so low.
"""

import sys
sys.path.append('src')

from agents.agent_types import create_contractor_from_paper_spec, create_sample_task
from decision.decision_engine import DecisionEngine

# Create test instances
decision_engine = DecisionEngine()
task = create_sample_task("financial_rag")
contractor = create_contractor_from_paper_spec("contractor_b")

print("="*80)
print("SKILL COMPATIBILITY DEBUG")
print("="*80)

print(f"\n📋 TASK DETAILS:")
print(f"Task Type: {task.task_type}")
print(f"Required Skills: {task.required_skills}")

print(f"\n🤖 CONTRACTOR DETAILS:")
print(f"Name: {contractor.name}")
print(f"Specialization: {contractor.specialization}")
print(f"Skills: {contractor.skills}")

# Calculate skill compatibility step by step
required_skills = set(task.required_skills)
contractor_skills = set(contractor.skills)

print(f"\n🔍 SKILL MATCHING CALCULATION:")
print(f"Required Skills Set: {required_skills}")
print(f"Contractor Skills Set: {contractor_skills}")

intersection = required_skills.intersection(contractor_skills)
union = required_skills.union(contractor_skills)

print(f"Intersection: {intersection}")
print(f"Union: {union}")

ontological_score = len(intersection) / len(union) if len(union) > 0 else 0.0
print(f"Ontological Score: {len(intersection)}/{len(union)} = {ontological_score:.3f}")

# Embedding similarity
embedding_score = decision_engine._calculate_embedding_similarity(contractor, task)
print(f"Embedding Score: {embedding_score:.3f}")

# Performance score
performance_score = contractor.performance_scores.get(task.task_type, 0.8)
print(f"Performance Score: {performance_score:.3f}")

# Final compatibility
compatibility = (0.3 * ontological_score + 
                0.5 * embedding_score + 
                0.2 * performance_score)

print(f"\n📊 FINAL CALCULATION:")
print(f"Compatibility = 0.3×{ontological_score:.3f} + 0.5×{embedding_score:.3f} + 0.2×{performance_score:.3f}")
print(f"Compatibility = {0.3 * ontological_score:.3f} + {0.5 * embedding_score:.3f} + {0.2 * performance_score:.3f}")
print(f"Compatibility = {compatibility:.3f}")

print(f"\n🎯 THRESHOLD CHECK:")
print(f"Skill Threshold: {decision_engine.skill_threshold}")
print(f"Passes Threshold: {'✅ YES' if compatibility >= decision_engine.skill_threshold else '❌ NO'}")

# Test all contractor types
print(f"\n🔄 ALL CONTRACTOR COMPATIBILITY:")
contractor_types = ["contractor_b", "contractor_c", "contractor_d", "contractor_e", "contractor_f", "contractor_g"]

for contractor_type in contractor_types:
    try:
        test_contractor = create_contractor_from_paper_spec(contractor_type)
        test_compatibility = decision_engine._assess_skill_compatibility(test_contractor, task)
        passes = "✅" if test_compatibility >= 0.7 else "❌"
        print(f"{passes} {test_contractor.name}: {test_compatibility:.3f}")
    except Exception as e:
        print(f"❌ {contractor_type}: Error - {e}")

print("="*80)