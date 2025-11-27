#!/bin/bash
#SBATCH --job-name=hierarchical_rne
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpua16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s.a.vink@uu.nl

###############################################################################
# Hierarchical Road Network Embedding (RNE) Experiment Script
#
# For NeurIPS submission on shortest-path distance approximation.
#
# Usage:
#   sbatch run_train.sbatch                                    # Default (standard config)
#   RNE_CONFIG=neurips sbatch run_train.sbatch                 # Full NeurIPS evaluation
#   RNE_CONFIG=quick sbatch run_train.sbatch                   # Quick test run
#   RNE_DATASETS="amazon dblp" sbatch run_train.sbatch         # Custom datasets
###############################################################################

set -e

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export GRAPHCUBES_SKIP_SSL=1

# Project directory - update for your environment
PROJECT_DIR="/storage/scratch/2427021/graph_surrogate"
cd "$PROJECT_DIR" || { echo "ERROR: Could not cd to $PROJECT_DIR"; exit 1; }

# Activate environment
if [ -f /storage/scratch/2427021/conda/etc/profile.d/conda.sh ]; then
    source /storage/scratch/2427021/conda/etc/profile.d/conda.sh
    conda activate graph_surrogate
fi

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment not found at .venv/bin/activate"
    exit 1
fi

python --version || { echo "ERROR: Python not found"; exit 1; }

###############################################################################
# Configuration with defaults
###############################################################################

# Default datasets from registry (good mix of sizes for NeurIPS)
# Small: deezer (~28K nodes), facebook (~22K nodes)
# Medium: amazon (~403K nodes), dblp (~317K nodes)
# Large: pokec (~1.6M nodes)
RNE_DATASETS="${RNE_DATASETS:-deezer facebook amazon dblp}"

# Configuration preset: quick, standard, or neurips
RNE_CONFIG="${RNE_CONFIG:-standard}"

# Run identifier (auto-generated if not set)
RNE_RUN_ID="${RNE_RUN_ID:-$(date +%Y-%m-%d_%H%M%S)}"

# Random seed for reproducibility
RNE_SEED="${RNE_SEED:-42}"

# Set to 1 to skip baseline model training (only train Hierarchical RNE)
RNE_ONLY="${RNE_ONLY:-0}"

###############################################################################
# Print configuration
###############################################################################

echo "============================================================================="
echo "Hierarchical Road Network Embedding (RNE) - NeurIPS Experiments"
echo "============================================================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Node:             $SLURM_NODELIST"
echo "GPUs:             $CUDA_VISIBLE_DEVICES"
echo "Start time:       $(date)"
echo "============================================================================="
echo "Configuration:"
echo "  Run ID:         $RNE_RUN_ID"
echo "  Datasets:       $RNE_DATASETS"
echo "  Config:         $RNE_CONFIG"
echo "  Seed:           $RNE_SEED"
echo "  RNE Only:       $RNE_ONLY"
echo "============================================================================="

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  (GPU info not available)"
echo "============================================================================="

###############################################################################
# Download datasets if needed
###############################################################################

echo "Checking/downloading datasets..."
for DATASET in $RNE_DATASETS; do
    echo "  - $DATASET"
    python -m datasets.registry download "$DATASET" 2>&1 || echo "    (may already exist or failed)"
done
echo "============================================================================="

###############################################################################
# Build and run command
###############################################################################

CMD="python -m scripts.run_rne_experiments"
CMD="$CMD --datasets $RNE_DATASETS"
CMD="$CMD --config $RNE_CONFIG"
CMD="$CMD --run-id $RNE_RUN_ID"
CMD="$CMD --seed $RNE_SEED"

if [ "$RNE_ONLY" = "1" ]; then
    CMD="$CMD --rne-only"
fi

if [ "$RNE_CONFIG" = "neurips" ]; then
    CMD="$CMD --neurips"
fi

echo "Running command:"
echo "  $CMD"
echo "============================================================================="

# Run experiments
$CMD

###############################################################################
# Summary
###############################################################################

echo ""
echo "============================================================================="
echo "Experiment completed!"
echo "============================================================================="
echo "End time:         $(date)"
echo "Results saved to: experiments/rne/$RNE_RUN_ID/"
echo "============================================================================="

# List output files
echo ""
echo "Output structure:"
ls -la "experiments/rne/$RNE_RUN_ID/" 2>/dev/null || echo "  (output directory not found)"
