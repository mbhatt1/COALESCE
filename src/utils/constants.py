"""
Constants and configuration parameters for the COALESCE simulation.

These values are based on the specifications and measurements from the paper.
"""

# Cost model constants (from paper Section 3.5)
ENERGY_COST_PER_KWH = 0.08  # USD per kWh
DATA_TRANSFER_COST_PER_GB = 0.12  # USD per GB
PROTOCOL_OVERHEAD_BASE_COST = 0.50  # USD base cost for protocol overhead
INTEGRATION_BASE_COST = 1.20  # USD for API and result integration
VERIFICATION_AUTOMATED_COST = 2.50  # USD for automated verification
VERIFICATION_MANUAL_COST = 3.80  # USD for manual verification
LATENCY_PENALTY_PER_MINUTE = 0.25  # USD penalty per minute over target

# Hardware specifications (from paper Table 1)
RTX_3080_PEAK_FLOPS = 29.77e12  # Peak FLOPS for RTX 3080
H100_PEAK_FLOPS = 1000e12  # Peak FLOPS for H100 (estimated)
RTX_4090_PEAK_FLOPS = 165e12  # Peak FLOPS for RTX 4090

# Memory requirements (from paper Section 4.1)
FINBERT_LARGE_MEMORY_GB = 1.3  # Model weights
KV_CACHE_PER_50K_DOCS_GB = 4.2  # KV cache for 50k documents
ACTIVATION_MEMORY_BASE_GB = 2.8  # Base activation memory

# Performance benchmarks (from paper Section 4.4)
DOCUMENTS_PER_KWH_LOCAL = 45  # Local execution efficiency
DOCUMENTS_PER_KWH_OPTIMIZED = 320  # Optimized contractor efficiency

# Market dynamics (from paper Section 4.3)
BUSINESS_HOURS_MULTIPLIER = 3.2  # Peak demand multiplier
QUARTERLY_EARNINGS_MULTIPLIER = 2.1  # Seasonal demand increase
MAINTENANCE_CAPACITY_REDUCTION = 0.20  # 20% capacity reduction during maintenance

# Security overhead (from paper Section 3.7)
CHACHA20_OVERHEAD_PERCENT = 0.023  # 2.3% computational overhead
ECDH_HANDSHAKE_MS = 15.7  # Average key exchange latency
SGX_ENCLAVE_OVERHEAD_PERCENT = 0.085  # 8.5% performance penalty
AUDIT_LOGGING_CPU_OVERHEAD = 0.008  # 0.8% CPU overhead
TOTAL_SECURITY_OVERHEAD = 0.128  # 12.8% combined overhead

# Decision algorithm parameters (from paper Section 3.6)
SKILL_COMPATIBILITY_THRESHOLD = 0.7  # θ_skill
TOPSIS_THRESHOLD = 0.6  # τ_threshold  
MIN_CONFIDENCE_THRESHOLD = 0.8  # ρ_min
LEARNING_RATE = 0.01  # η for weight updates

# Skill compatibility weights (from paper Eq. 11)
ONTOLOGICAL_WEIGHT = 0.3  # α
EMBEDDING_WEIGHT = 0.5  # β
PERFORMANCE_WEIGHT = 0.2  # γ

# EWMA calibration (from paper Eq. 10)
EWMA_LAMBDA_MIN = 0.1
EWMA_LAMBDA_MAX = 0.3
EWMA_LAMBDA_DEFAULT = 0.2

# Market stability indicators (from paper Section 5.3)
PRICE_VOLATILITY_THRESHOLD = 0.5  # Coefficient of variation threshold
HHI_MONOPOLY_THRESHOLD = 2500  # Herfindahl-Hirschman Index threshold
BID_ASK_SPREAD_THRESHOLD = 0.15  # 15% spread threshold

# System performance targets (from paper Section 4.4)
TARGET_COST_REDUCTION = 0.64  # 64% cost reduction target
TARGET_TIME_SAVINGS = 0.93  # 93% time savings target
TARGET_THROUGHPUT_IMPROVEMENT = 5.5  # 550% throughput improvement
TARGET_ENERGY_EFFICIENCY = 6.11  # 611% efficiency improvement

# Simulation parameters
DEFAULT_SIMULATION_DURATION_DAYS = 30
DEFAULT_TIME_STEP_SECONDS = 3600  # 1 hour time steps
DEFAULT_POISSON_LAMBDA = 4.2  # Tasks per minute during peak
DEFAULT_NUM_CLIENT_AGENTS = 50
DEFAULT_NUM_CONTRACTOR_AGENTS = 200

# Network simulation parameters
DEFAULT_BANDWIDTH_MBPS = 1000
DEFAULT_LATENCY_MS = 15.0
DEFAULT_RELIABILITY = 0.99

# Task generation parameters
TASK_TYPE_DISTRIBUTION = {
    'financial_rag': 0.4,
    'risk_assessment': 0.3,
    'sentiment_analysis': 0.2,
    'portfolio_optimization': 0.1
}

COMPLEXITY_FACTOR_RANGE = (0.5, 2.0)
VALUE_RANGE = (50.0, 500.0)
DATA_SIZE_RANGE_GB = (1.0, 20.0)

# Contractor distribution (from paper Table 1)
CONTRACTOR_TYPE_DISTRIBUTION = {
    'contractor_b': 0.152,  # Premium GPU Specialist
    'contractor_c': 0.228,  # High-Throughput CPU
    'contractor_d': 0.315,  # Budget Provider
    'contractor_e': 0.087,  # Edge Specialist
    'contractor_f': 0.183,  # Hybrid Cloud
    'contractor_g': 0.035   # Quantum-Enhanced
}

# Reliability score ranges by contractor type
RELIABILITY_RANGES = {
    'contractor_b': (0.98, 0.99),
    'contractor_c': (0.90, 0.93),
    'contractor_d': (0.70, 0.80),
    'contractor_e': (0.82, 0.89),
    'contractor_f': (0.93, 0.96),
    'contractor_g': (0.87, 0.91)
}

# Performance variation ranges
LATENCY_VARIATION_PERCENT = 0.15  # ±15% variation
COST_VARIATION_PERCENT = 0.10  # ±10% variation
DEMAND_VARIATION_PERCENT = 0.25  # ±25% variation

# Report generation parameters
REPORT_CONFIDENCE_INTERVALS = [0.90, 0.95, 0.99]
BOOTSTRAP_SAMPLES = 1000
CHART_DPI = 300
CHART_STYLE = 'seaborn-v0_8'

# File paths
CONFIG_DIR = "config"
OUTPUT_DIR = "output"
LOGS_DIR = "logs"
REPORTS_DIR = "reports"
DATA_DIR = "data"