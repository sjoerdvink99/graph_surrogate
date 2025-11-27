# Reproducibility Guide for GraphSurrogate

This document provides comprehensive instructions for reproducing all experiments in the GraphSurrogate paper, including the extended experiments addressing potential reviewer concerns.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Quick Start](#quick-start)
3. [Main Experiments](#main-experiments)
4. [Extended Experiments](#extended-experiments)
5. [Ablation Studies](#ablation-studies)
6. [Datasets](#datasets)
7. [Expected Results](#expected-results)

## Environment Setup

### Requirements

- Python 3.12+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/GraphSurrogate.git
cd GraphSurrogate

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Verify Installation

```bash
# Run tests
pytest tests/test_core.py -v

# Expected: All 50+ tests should pass
```

## Quick Start

Run a quick validation experiment:

```bash
# Quick test (3 datasets, 1 seed, ~10 minutes)
uv run neurips-study --quick --output-dir experiments/quick_test
```

This produces:

- Model checkpoints in `experiments/quick_test/mlp/`
- Results JSON files
- LaTeX tables in `experiments/quick_test/tables/`
- Figures in `experiments/quick_test/figures/`

## Main Experiments

### Full NeurIPS Study

Run the complete experimental study:

```bash
# Full study (8 datasets, 3 seeds, ~8-12 hours)
uv run neurips-study --output-dir experiments/neurips_final
```

### Configuration Options

```bash
# Custom datasets
uv run neurips-study --datasets bron github deezer --output-dir experiments/custom

# Different seeds
uv run neurips-study --seeds 1 2 3 4 5 --output-dir experiments/more_seeds

# Skip GNN baseline (faster)
uv run neurips-study --skip-gnn --output-dir experiments/no_gnn
```

### Output Structure

```
experiments/neurips_final/
├── config.json                 # Experiment configuration
├── all_results.json           # Raw results
├── aggregated_results.json    # Aggregated metrics with CIs
├── mlp/                       # MLP model results
│   └── <dataset>/
│       └── seed_<N>/
│           ├── model.pt       # Trained model
│           ├── results.json   # Evaluation metrics
│           └── encoder_config.json
├── gnn/                       # GNN baseline results
├── baselines/                 # Non-learned baselines
├── ablations/                 # Ablation study results
├── tables/                    # LaTeX tables
│   ├── table_accuracy.tex
│   ├── table_latency.tex
│   └── table_ablation.tex
└── figures/                   # Publication figures
    ├── pareto_frontier.pdf
    ├── comparison_bars.pdf
    └── ablations.pdf
```

## Extended Experiments

### Gap 2: Scalability (Large-Scale Evaluation)

```bash
# Run scalability experiments
uv run neurips-extended --gaps 2 --output-dir experiments/scalability

# Expected runtime: ~2-4 hours
# Tests graphs from 10K to 1M nodes
```

### Gap 4: Cross-Graph Generalization

```bash
# Test transfer learning across graphs
uv run neurips-extended --gaps 4 --output-dir experiments/transfer

# Trains on one graph, evaluates on variants
```

### Gap 5: Cardinality Estimation Comparison

```bash
# Compare with database cardinality methods
uv run neurips-extended --gaps 5 --output-dir experiments/cardinality

# Compares: Enhanced Histogram, NeuroCard-inspired, DeepDB-inspired
```

### Gap 7: Failure Case Analysis

```bash
# Comprehensive failure analysis
uv run neurips-extended --gaps 7 --output-dir experiments/failures

# Outputs:
# - failure_report.json with categorized failures
# - failure_analysis.pdf with visualizations
# - Actionable recommendations
```

### Gap 8: Comprehensive Ablations

```bash
# Extended ablation studies
uv run neurips-extended --gaps 8 --output-dir experiments/ablations

# Tests:
# - Architecture (layers, dimensions)
# - Training size sensitivity
# - Hyperparameter sensitivity
```

### Gap 9: Uncertainty Quantification

```bash
# Uncertainty estimation experiments
uv run neurips-extended --gaps 9 --output-dir experiments/uncertainty

# Evaluates:
# - MC Dropout uncertainty
# - Calibration analysis
# - Uncertainty-error correlation
```

### Gap 10: Incremental Updates

```bash
# Graph change impact analysis
uv run neurips-extended --gaps 10 --output-dir experiments/incremental

# Simulates graph evolution and measures model degradation
```

### Run All Extended Experiments

```bash
# All gaps (recommended for final submission)
uv run neurips-extended --output-dir experiments/all_gaps

# Quick test mode
uv run neurips-extended --quick --output-dir experiments/all_gaps_quick
```

## Ablation Studies

### Architecture Ablations

The study tests:

- **Number of layers**: 1, 2, 3, 4, 5
- **Latent dimension**: 8, 16, 32, 64, 128
- **Hidden dimension**: 64, 128, 256, 512

### Training Ablations

- **Training size**: 10%, 25%, 50%, 100% of data
- **Dropout rate**: 0.0, 0.05, 0.1, 0.2, 0.3

### Encoding Ablations

- **No degree bins**: Removes degree-based encoding
- **No radius**: Single radius encoding
- **No attribute filter**: Removes filter encoding

## Datasets

### Download Datasets

```bash
# Download all datasets
uv run download-dataset download bron
uv run download-dataset download github
uv run download-dataset download deezer
# ... etc

# List available datasets
uv run download-dataset list
```

### Dataset Statistics

| Dataset  | Nodes | Edges | Description            |
| -------- | ----- | ----- | ---------------------- |
| bron     | 30K   | 150K  | Threat intelligence    |
| github   | 37K   | 578K  | Developer social       |
| deezer   | 28K   | 185K  | Music social           |
| facebook | 22K   | 340K  | Page network           |
| amazon   | 403K  | 3.4M  | Co-purchasing          |
| dblp     | 317K  | 1M    | Academic collaboration |
| youtube  | 1.1M  | 3M    | Large-scale social     |
| orkut    | 3M    | 117M  | Very large-scale       |

### Custom Datasets

To add a custom dataset:

```python
from datasets.registry import load_dataset
from training.graph_loader import load_graph

# Load from edge list
G, metadata = load_graph("path/to/edges.txt", name="my_dataset")

# Ensure nodes have attributes
for node in G.nodes():
    G.nodes[node]["node_type"] = "default"
```

## Expected Results

### Main Results (Table 1)

Expected ranges for GraphSurrogate vs baselines:

| Metric       | GraphSurrogate | GNN     | Histogram |
| ------------ | -------------- | ------- | --------- |
| Count MAE    | 5-15           | 15-30   | 30-50     |
| Dist MAE     | 0.3-0.8        | 0.5-1.2 | 1.5-2.5   |
| Latency (ms) | 0.02-0.05      | 1-5     | 0.1-0.5   |

### Scalability (Gap 2)

Expected speedup over graph traversal:

- 10K nodes: 10-50x
- 100K nodes: 100-500x
- 1M nodes: 1000x+

### Cross-Graph Transfer (Gap 4)

Expected accuracy retention:

- Same-structure variant: 90-100%
- Denser graph (+50% edges): 70-85%
- Sparser graph (-50% edges): 60-80%

### Uncertainty Calibration (Gap 9)

Expected uncertainty-error correlation: >0.5 indicates well-calibrated model

## Computational Resources

### Minimum Requirements

- 16 GB RAM
- 8 CPU cores
- ~50 GB disk space

### Recommended (for full study)

- 32+ GB RAM
- GPU with 8+ GB VRAM
- 100 GB disk space

### Estimated Runtimes

| Experiment          | CPU Only  | With GPU  |
| ------------------- | --------- | --------- |
| Quick test          | ~30 min   | ~10 min   |
| Full study          | ~24 hours | ~8 hours  |
| Extended (all gaps) | ~48 hours | ~16 hours |

## Troubleshooting

### Memory Issues

```bash
# Reduce batch size
uv run neurips-study --batch-size 128 --output-dir experiments/lowmem

# Use streaming data loading
uv run train --streaming --num-workers 2
```

### SSL Certificate Errors (HPC Clusters)

```bash
export GRAPHCUBES_SKIP_SSL=1
uv run download-dataset download bron
```

### Missing Dependencies

```bash
# Install PyTorch Geometric for GNN baseline
pip install torch-geometric

# Install optional visualization dependencies
pip install seaborn plotly
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{graphsurrogate2025,
  title={GraphSurrogate: Learning to Answer Graph Aggregate Queries in Constant Time},
  author={...},
  booktitle={NeurIPS},
  year={2025}
}
```

## License

MIT License. See LICENSE file for details.
