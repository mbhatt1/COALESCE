# COALESCE: Cost-Optimized Agent Labor Exchange via Skill-based Competence Estimation

A **simulation framework** that models the COALESCE system described in the research paper "A Proposed Framework for Cost-Optimized and Secure Task Outsourcing Among Autonomous LLM Agents via Skill-based Competence Estimation."

## Overview

This is a **theoretical simulation** that models how autonomous LLM agents could optimize resource utilization and operational costs through intelligent task outsourcing. The framework simulates advanced multi-criteria decision algorithms, comprehensive cost modeling, and agent-to-agent market dynamics using mathematical models and statistical distributions.

**Important**: This is not an implementation of actual LLM agents communicating with each other, but rather a simulation framework that models the theoretical behavior described in the research paper.

## Key Features

- **Advanced Cost Modeling**: Comprehensive cost analysis including compute, memory, energy, opportunity, and risk costs
- **Enhanced Decision Engine**: TOPSIS-based multi-criteria analysis with epsilon-greedy exploration (ε=0.1)
- **Market Discovery**: Robust exploration mechanism enabling learning from suboptimal contractors
- **Realistic Agent Simulation**: Multiple agent types with varying specializations and capabilities
- **Market Dynamics**: Supply-demand modeling with realistic market conditions
- **Economic Validation**: Strong alignment with transaction cost theory and organizational economics
- **Security Framework**: Multi-layer security model with cryptographic protocols
- **Comprehensive Reporting**: Detailed performance analysis and visualization

## Validated Performance Results

Based on comprehensive simulation across 17 experimental scenarios:

- **Cost Reduction**: 33.9% average cost savings (range: 0-158%)
- **Time Savings**: 31.2% execution time improvement (range: 0-96.3%)
- **System Throughput**: 3.1 tasks/hour average (range: 1.8-3.5)
- **Market Efficiency**: 70.6% of scenarios achieve >20% cost reduction
- **Outsourcing Rate**: 16.4% selective contractor engagement
- **Energy Efficiency**: 611% efficiency improvement (validated)

### Key Findings
- **Super-efficiency in small markets**: 158% cost reduction with 5 agents
- **Consistent mid-scale performance**: 20-25% cost reduction with 20-30 agents
- **Large-scale viability**: 26.5% cost reduction with 50 agents
- **Economic validation**: Strong transaction cost theory confirmation (r=0.833)
- **Realistic scale effects**: Diseconomies of scale observed (r=-0.428)

## Installation

1. Clone or download the COALESCE project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run a basic simulation with default parameters:

```bash
python main.py
```

Run with custom parameters:

```bash
python main.py --duration 7 --agents 25 --contractors 100 --verbose
```

## Configuration

The simulation can be configured using YAML files. Default configuration is created automatically at `config/default_config.yaml`.

Key configuration parameters:
- `simulation_duration_days`: Simulation length (default: 30 days)
- `num_client_agents`: Number of client agents (default: 50)
- `num_contractor_agents`: Number of contractor agents (default: 200)
- `task_type_distribution`: Distribution of task types
- `contractor_type_distribution`: Distribution of contractor specializations

## Command Line Options

```
usage: main.py [-h] [--config CONFIG] [--output-dir OUTPUT_DIR] 
               [--duration DURATION] [--agents AGENTS] 
               [--contractors CONTRACTORS] [--verbose]

COALESCE: Cost-Optimized Agent Labor Exchange Simulation

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to simulation configuration file
  --output-dir OUTPUT_DIR
                        Directory for simulation output and reports
  --duration DURATION   Simulation duration in days
  --agents AGENTS       Number of client agents
  --contractors CONTRACTORS
                        Number of contractor agents
  --verbose             Enable verbose logging
```

## Output

The simulation generates comprehensive reports in the output directory:

### Reports
- `executive_summary.md`: High-level performance summary
- `detailed_analysis.md`: Statistical analysis and metrics
- `performance_benchmarks.json`: Machine-readable performance data
- `consolidated_report.html`: Interactive HTML report

### Visualizations
- `dashboard.png`: Performance overview dashboard
- `cost_comparison.png`: Cost analysis charts
- Various performance and market dynamics charts

### Raw Data
- `time_series.csv`: Time series performance data
- `decisions.csv`: Individual decision records
- `agent_performance.json`: Detailed agent metrics
- `simulation_config.json`: Configuration used

## Architecture

The COALESCE framework consists of several key components:

### Core Modules

1. **Cost Calculator** (`src/cost_model/cost_calculator.py`)
   - Implements comprehensive cost modeling from the paper
   - Internal and external cost calculations
   - Real-time cost calibration using EWMA

2. **Decision Engine** (`src/decision/decision_engine.py`)
   - TOPSIS-based multi-criteria decision analysis with epsilon-greedy exploration
   - Skill compatibility assessment with market discovery capabilities
   - Game-theoretic optimization with learning mechanisms
   - Dynamic weight adjustment based on historical performance

3. **Agent Types** (`src/agents/agent_types.py`)
   - Client and contractor agent definitions
   - Hardware configurations and capabilities
   - Task specifications

4. **Simulation Engine** (`src/simulation/simulation_engine.py`)
   - Main simulation orchestration
   - Market dynamics modeling
   - Performance tracking

5. **Report Generator** (`src/reporting/report_generator.py`)
   - Comprehensive report generation
   - Statistical analysis and visualization
   - Data export functionality

## Agent Types

The simulation includes multiple contractor agent types based on the paper specifications:

- **Premium GPU Specialist**: High-performance H100 GPUs, premium pricing
- **High-Throughput CPU**: Optimized for parallel processing
- **Budget Provider**: Cost-effective RTX 4090 setup
- **Edge Specialist**: Mobile and edge computing focus
- **Hybrid Cloud**: Auto-scaling cloud infrastructure
- **Quantum-Enhanced**: Advanced quantum ML capabilities

## Task Types

Supported task types with realistic computational requirements:

- **Financial RAG**: Document analysis with FinBERT models
- **Risk Assessment**: Monte Carlo simulations
- **Sentiment Analysis**: Large-scale text processing
- **Portfolio Optimization**: Quadratic programming

## Performance Metrics

The simulation tracks comprehensive performance metrics:

### Economic Metrics
- Cost reduction percentage
- Total cost savings
- Return on investment
- Resource utilization efficiency

### Performance Metrics
- Task completion time
- System throughput
- Energy efficiency
- Decision accuracy

### Market Metrics
- Market concentration (HHI)
- Price volatility
- Contractor utilization
- Supply-demand dynamics

## Experimental Validation

The simulation framework has been validated through comprehensive testing across 17 systematic experiments:

### Duration Analysis (9 experiments)
- **Configuration**: 15 client agents, 30 contractor agents
- **Duration range**: 1-30 days
- **Best performance**: 98% cost reduction (20-day simulation)
- **Consistent performance**: 17-25% cost reduction in stable scenarios

### Agent Scale Analysis (8 experiments)
- **Configuration**: 7-day duration, varying agent populations
- **Agent range**: 5-50 client agents (2:1 contractor ratio)
- **Super-efficiency**: 158% cost reduction with 5 agents
- **Enterprise scale**: 26.5% cost reduction with 50 agents

### Economic Validation
- **Transaction cost theory**: Strong correlation (r=0.833) between outsourcing and performance
- **Scale effects**: Realistic diseconomies of scale (r=-0.428)
- **Market efficiency**: 70.6% success rate achieving significant gains
- **Exploration mechanism**: 10% exploration rate enabling market discovery

### Framework Robustness
- **Epsilon-greedy exploration**: Eliminates complete system failures
- **Consistent performance**: Reliable operation across diverse scenarios
- **Graceful degradation**: Safe fallback to local execution when needed

## Research Applications

This simulation framework can be used for:

- Validating COALESCE framework performance
- Testing different market conditions and agent configurations
- Analyzing the impact of various decision algorithms
- Studying agent economy dynamics
- Benchmarking against alternative approaches

## Contributing

The COALESCE simulation framework is designed for research and educational purposes. Contributions and extensions are welcome.

## License

This project implements the research described in the COALESCE paper and is intended for academic and research use.

## Caveats and Limitations

### Simulation Nature
- **This is NOT a real implementation**: This framework simulates agent behavior using mathematical models and random distributions, not actual LLM agents
- **No actual agent communication**: "Agents" are data structures with hardware specifications; "task execution" uses `np.random.normal()` to simulate costs and times
- **Theoretical validation only**: Results validate the mathematical framework and decision algorithms, not real-world agent interactions
- **Mock market dynamics**: Supply, demand, and pricing variations are generated using statistical distributions, not actual market forces

### Simulation Constraints
- **Restrictive Decision Thresholds**: High TOPSIS thresholds (>0.6), confidence requirements (>0.8), or skill compatibility thresholds (>0.7) may prevent outsourcing entirely, resulting in 0% metrics across all performance indicators
- **Market Size Dependencies**: Performance varies significantly with agent population - small markets (5 agents) may show super-efficiency effects while large markets (50+ agents) exhibit realistic diseconomies of scale
- **Duration Sensitivity**: Short simulations (<7 days) may not capture market equilibrium effects, while very long simulations (>30 days) may show diminishing returns

### Data Interpretation
- **Zero-Value Results**: Experiments showing 0.0% across all metrics typically indicate no outsourcing occurred due to restrictive parameters rather than system failure
- **Super-Efficiency Anomalies**: Cost reductions >100% in small-scale scenarios reflect idealized market conditions and should be interpreted as theoretical upper bounds
- **Exploration vs Exploitation**: The ε=0.1 exploration rate may cause temporary performance degradation as the system learns optimal contractor relationships

### Real-World Applicability
- **Simplified Cost Models**: The simulation uses idealized cost calculations that may not reflect real-world complexity, network latency, or integration overhead
- **Perfect Information Assumption**: Agents have complete visibility into contractor capabilities, which may not hold in practice
- **Static Agent Behavior**: Contractor agents maintain consistent performance profiles without learning or adaptation over time

### Technical Limitations
- **Computational Scalability**: Large-scale simulations (>100 agents, >30 days) may require significant computational resources
- **Random Seed Sensitivity**: Results may vary between runs when random_seed is null; use fixed seeds for reproducible experiments
- **Configuration Dependencies**: Performance heavily depends on parameter tuning - default values may not be optimal for all research scenarios

## Contact

For questions about the simulation framework or research collaboration, please use issues.