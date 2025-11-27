# GraphCubes

Neural surrogate for graph aggregate queries. Approximates k-hop neighborhood counts and shortest-path distances with low latency.

## Setup

```bash
uv sync
```

## Usage

```bash
# Train on a dataset
uv run python -m training.train --dataset deezer --output-dir output/deezer

# Evaluate
uv run python -m scripts.evaluate --model-dir output/deezer

# Generate figures
uv run python -m scripts.visualizations --output-dir output/deezer
```

## Datasets

```bash
uv run python -m datasets.registry list
uv run python -m datasets.registry download bron
```

| Dataset  | Nodes | Edges |
| -------- | ----- | ----- |
| bron     | ~30K  | ~150K |
| github   | ~37K  | ~580K |
| deezer   | ~28K  | ~185K |
| facebook | ~22K  | ~340K |
| amazon   | ~400K | ~3.4M |
| dblp     | ~317K | ~1M   |
| youtube  | ~1.1M | ~3M   |
| reddit   | ~55K  | ~860K |

## Large graphs

```bash
uv run python -m training.train --dataset amazon --streaming --num-train 500000
```
