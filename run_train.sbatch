#!/bin/bash
#SBATCH --job-name=graph_surrogate
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpua16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s.a.vink@uu.nl

# =============================================================================
# Graph Surrogate - Full Experiment Pipeline
# =============================================================================
# This script runs the complete experiment: training, evaluation, latency
# benchmarking, and figure generation across multiple datasets.
#
# Usage:
#   sbatch run_train.sbatch                      # Run with defaults
#   DATASETS="deezer github" sbatch run_train.sbatch
#   RUN_ID="my_experiment" sbatch run_train.sbatch
#   KUZU=1 sbatch run_train.sbatch               # Include Kuzu benchmark
# =============================================================================

set -e  # Exit on error

# Set up environment
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# Navigate to project directory
cd /storage/scratch/2427021/graph_surrogate

# Activate conda environment
source /storage/scratch/2427021/conda/etc/profile.d/conda.sh
conda activate graph_surrogate

# Use the uv-created venv
source .venv/bin/activate

# =============================================================================
# Configuration - Override these with environment variables
# =============================================================================
DATASETS="${DATASETS:-deezer github facebook bron amazon dblp}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H%M%S)}"
NUM_TRAIN="${NUM_TRAIN:-100000}"
NUM_VAL="${NUM_VAL:-10000}"
NUM_TEST="${NUM_TEST:-10000}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LATENCY_QUERIES="${LATENCY_QUERIES:-2000}"
STREAMING="${STREAMING:-0}"
KUZU="${KUZU:-0}"

# =============================================================================
# Print configuration
# =============================================================================
echo "============================================================================="
echo "Graph Surrogate - Full Experiment Pipeline"
echo "============================================================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Node:             $SLURM_NODELIST"
echo "GPUs:             $CUDA_VISIBLE_DEVICES"
echo "Start time:       $(date)"
echo "============================================================================="
echo "Configuration:"
echo "  Run ID:         $RUN_ID"
echo "  Datasets:       $DATASETS"
echo "  Training:       $NUM_TRAIN samples, $EPOCHS epochs, batch=$BATCH_SIZE"
echo "  Validation:     $NUM_VAL samples"
echo "  Test:           $NUM_TEST samples"
echo "  Latency:        $LATENCY_QUERIES queries"
echo "  Streaming:      $STREAMING"
echo "  Kuzu:           $KUZU"
echo "============================================================================="

# Print GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "============================================================================="

# =============================================================================
# Build command
# =============================================================================
CMD="python -m scripts.run_experiments"
CMD="$CMD --datasets $DATASETS"
CMD="$CMD --run-id $RUN_ID"
CMD="$CMD --num-train $NUM_TRAIN"
CMD="$CMD --num-val $NUM_VAL"
CMD="$CMD --num-test $NUM_TEST"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --latency-queries $LATENCY_QUERIES"

if [ "$STREAMING" = "1" ]; then
    CMD="$CMD --streaming"
fi

if [ "$KUZU" = "1" ]; then
    CMD="$CMD --kuzu"
fi

# =============================================================================
# Run experiment
# =============================================================================
echo "Running command:"
echo "  $CMD"
echo "============================================================================="

$CMD

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================================="
echo "Experiment completed!"
echo "============================================================================="
echo "End time:         $(date)"
echo "Results saved to: experiments/$RUN_ID/"
echo ""
echo "Output structure:"
echo "  experiments/$RUN_ID/"
echo "  ├── config.json              # Experiment configuration"
echo "  ├── results_summary.json     # Combined results"
echo "  ├── figures/"
echo "  │   ├── comparison.pdf       # Cross-dataset comparison"
echo "  │   ├── table_accuracy.tex   # LaTeX table for paper"
echo "  │   ├── table_latency.tex    # LaTeX table for paper"
echo "  │   └── table_throughput.tex # LaTeX table for paper"
echo "  └── {dataset}/               # Per-dataset outputs"
echo "      ├── model.pt"
echo "      ├── evaluation_results.json"
echo "      ├── latency_benchmark.json"
echo "      └── figures/"
echo "============================================================================="
