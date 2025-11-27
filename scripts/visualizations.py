import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from model.encoder import EncoderConfig, QueryEncoder
from model.network import GraphSurrogate

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'axes.grid': False,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'legend.borderaxespad': 0.5,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

COLUMN_WIDTH = 3.5
FULL_WIDTH = 7.0

COLORS = {
    'primary': '#2171b5',
    'secondary': '#cb181d',
    'tertiary': '#238b45',
    'quaternary': '#6a51a3',
    'light_blue': '#9ecae1',
    'light_red': '#fc9272',
    'gray': '#636363',
    'light_gray': '#bdbdbd',
}


def load_results(output_dir: Path) -> tuple[dict, dict, list]:
    eval_path = output_dir / 'evaluation_results.json'
    history_path = output_dir / 'training_history.json'
    queries_path = output_dir / 'test_queries.json'

    eval_results = {}
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)

    history = {}
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    queries = []
    if queries_path.exists():
        with open(queries_path) as f:
            queries = json.load(f)

    return eval_results, history, queries


def generate_predictions(output_dir: Path, device: str = "cpu") -> tuple[np.ndarray, np.ndarray, list]:
    test_data_path = output_dir / "test_data.pt"
    if not test_data_path.exists():
        return None, None, None

    data = torch.load(test_data_path, weights_only=True)
    X_test = data["X"]
    y_test = data["y"]
    qt_test = data["qt"]

    config = EncoderConfig.load(output_dir / "encoder_config.json")
    with open(output_dir / "model_info.json") as f:
        model_info = json.load(f)

    model = GraphSurrogate(
        input_dim=model_info["input_dim"],
        hidden_dim=model_info.get("hidden_dim", 128),
        latent_dim=model_info.get("latent_dim", 32),
        num_layers=model_info.get("num_layers", 3),
        dropout=model_info.get("dropout", 0.1),
    )
    state_dict = torch.load(output_dir / "model.pt", weights_only=True, map_location=device)
    # Handle torch.compile() prefix - strip "_orig_mod." if present
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        count_pred, dist_pred, _ = model(X_test)

    predictions = torch.where(qt_test == 0, count_pred, dist_pred).numpy()
    ground_truths = y_test.numpy()
    query_types = qt_test.numpy()

    return predictions, ground_truths, query_types


def create_metrics_table(
    eval_results: dict,
    save_path: Path,
    dataset_name: str = "Dataset",
):
    save_path.parent.mkdir(exist_ok=True)

    accuracy = eval_results.get('accuracy', {})
    latency = eval_results.get('latency', {})
    memory = eval_results.get('memory', {})
    model_info = eval_results.get('model', {})

    latex = r"""\begin{table}[t]
\centering
\caption{Model performance on %s.}
\label{tab:performance_%s}
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
\multicolumn{2}{@{}l}{\textit{Count Queries}} \\
\quad MAE & %.2f \\
\quad RMSE & %.2f \\
\quad MAPE & %.1f\%% \\
\quad Pearson r & %.3f \\
\midrule
\multicolumn{2}{@{}l}{\textit{Distance Queries}} \\
\quad MAE & %.2f \\
\quad RMSE & %.2f \\
\quad Accuracy ($\pm$1 hop) & %.1f\%% \\
\midrule
\multicolumn{2}{@{}l}{\textit{Latency (ms)}} \\
\quad Model (mean) & %.3f \\
\quad Model (p99) & %.3f \\
\quad Ground truth (mean) & %.3f \\
\quad Speedup & %.1f$\times$ \\
\midrule
\multicolumn{2}{@{}l}{\textit{Model}} \\
\quad Parameters & %s \\
\quad Size & %.1f KB \\
\bottomrule
\end{tabular}
\end{table}""" % (
        dataset_name,
        dataset_name.lower().replace(" ", "_"),
        accuracy.get('count_mae', 0),
        accuracy.get('count_rmse', 0),
        accuracy.get('count_mape', 0),
        accuracy.get('count_pearson_r', 0),
        accuracy.get('dist_mae', 0),
        accuracy.get('dist_rmse', 0),
        accuracy.get('dist_accuracy_1hop', 0),
        latency.get('model_mean_ms', 0),
        latency.get('model_p99_ms', 0),
        latency.get('db_mean_ms', 0),
        latency.get('speedup_mean', 0),
        f"{model_info.get('num_params', 0):,}",
        memory.get('model_size_kb', 0),
    )

    with open(save_path, 'w') as f:
        f.write(latex)

    print(f"Saved metrics table to {save_path}")
    return latex


def plot_training_curves(history: dict, save_path: Path):
    save_path.parent.mkdir(exist_ok=True)

    if not history:
        print("No training history available")
        return

    epochs = np.arange(1, len(history.get('train_loss', [])) + 1)
    if len(epochs) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.2))

    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], color=COLORS['primary'], label='Train', linewidth=1.0)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('(a) Training Loss')
    ax1.legend(loc='upper right')
    ax1.set_xlim(1, len(epochs))

    ax2 = axes[1]
    if 'val_combined_mae' in history:
        ax2.plot(epochs, history['val_combined_mae'], color=COLORS['secondary'], label='Combined', linewidth=1.0)
    if 'val_count_mae' in history:
        ax2.plot(epochs, history['val_count_mae'], color=COLORS['primary'], label='Count', linewidth=1.0, alpha=0.7)
    if 'val_dist_mae' in history:
        ax2.plot(epochs, history['val_dist_mae'], color=COLORS['tertiary'], label='Distance', linewidth=1.0, alpha=0.7)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('(b) Validation MAE')
    ax2.set_xlim(1, len(epochs))
    ax2.legend(loc='upper right')

    if 'val_combined_mae' in history and history['val_combined_mae']:
        best_mae = min(history['val_combined_mae'])
        best_epoch = history['val_combined_mae'].index(best_mae) + 1
        ax2.axhline(y=best_mae, color=COLORS['light_gray'], linestyle='--', linewidth=0.75)
        ax2.annotate(f'Best: {best_mae:.1f}',
                     xy=(best_epoch, best_mae),
                     xytext=(best_epoch + 10, best_mae * 1.1),
                     fontsize=7, color=COLORS['gray'])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300)
    plt.close()

    print(f"Saved training curves to {save_path}")


def plot_predictions_scatter(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    query_types: np.ndarray,
    save_path: Path,
):
    save_path.parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0))

    ax1 = axes[0]
    count_mask = query_types == 0
    if count_mask.any():
        ax1.scatter(ground_truths[count_mask], predictions[count_mask],
                    c=COLORS['primary'], alpha=0.4, s=8, edgecolors='none')
        max_val = max(ground_truths[count_mask].max(), predictions[count_mask].max())
        ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=0.75, alpha=0.5)
        ax1.set_xlim(0, max_val * 1.05)
        ax1.set_ylim(0, max_val * 1.05)

        corr = np.corrcoef(ground_truths[count_mask], predictions[count_mask])[0, 1]
        ax1.text(0.95, 0.05, f'r = {corr:.3f}', transform=ax1.transAxes,
                 ha='right', va='bottom', fontsize=8, color=COLORS['gray'])

    ax1.set_xlabel('Ground Truth')
    ax1.set_ylabel('Predicted')
    ax1.set_title('(a) Count Queries')
    ax1.set_aspect('equal')

    ax2 = axes[1]
    dist_mask = query_types == 1
    if dist_mask.any():
        ax2.scatter(ground_truths[dist_mask], predictions[dist_mask],
                    c=COLORS['secondary'], alpha=0.4, s=8, edgecolors='none')
        max_val = max(ground_truths[dist_mask].max(), predictions[dist_mask].max(), 7)
        ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=0.75, alpha=0.5)
        ax2.set_xlim(-0.5, max_val + 0.5)
        ax2.set_ylim(-0.5, max_val + 0.5)

        mae = np.mean(np.abs(predictions[dist_mask] - ground_truths[dist_mask]))
        ax2.text(0.95, 0.05, f'MAE = {mae:.2f}', transform=ax2.transAxes,
                 ha='right', va='bottom', fontsize=8, color=COLORS['gray'])

    ax2.set_xlabel('Ground Truth')
    ax2.set_ylabel('Predicted')
    ax2.set_title('(b) Distance Queries')
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300)
    plt.close()

    print(f"Saved predictions scatter to {save_path}")


def plot_error_distribution(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    query_types: np.ndarray,
    save_path: Path,
):
    save_path.parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.2))

    ax1 = axes[0]
    count_mask = query_types == 0
    if count_mask.any():
        count_errors = predictions[count_mask] - ground_truths[count_mask]
        ax1.hist(count_errors, bins=50, color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=0.3)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.75)
        ax1.axvline(x=np.mean(count_errors), color=COLORS['secondary'], linestyle='-', linewidth=1.0)
        ax1.text(0.95, 0.95, f'Mean: {np.mean(count_errors):.1f}\nStd: {np.std(count_errors):.1f}',
                 transform=ax1.transAxes, ha='right', va='top', fontsize=7,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(a) Count Query Errors')

    ax2 = axes[1]
    dist_mask = query_types == 1
    if dist_mask.any():
        dist_errors = predictions[dist_mask] - ground_truths[dist_mask]
        ax2.hist(dist_errors, bins=30, color=COLORS['secondary'], alpha=0.8, edgecolor='white', linewidth=0.3)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.75)
        ax2.text(0.95, 0.95, f'Mean: {np.mean(dist_errors):.2f}\nStd: {np.std(dist_errors):.2f}',
                 transform=ax2.transAxes, ha='right', va='top', fontsize=7,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('(b) Distance Query Errors')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300)
    plt.close()

    print(f"Saved error distribution to {save_path}")


def load_all_dataset_results(results_dirs: list[Path], dataset_names: list[str]) -> list[dict]:
    """Load evaluation and latency results from all datasets."""
    all_results = []

    for result_dir, name in zip(results_dirs, dataset_names):
        data = {"name": name, "dir": result_dir}

        eval_path = result_dir / 'evaluation_results.json'
        if eval_path.exists():
            with open(eval_path) as f:
                data["eval"] = json.load(f)

        latency_path = result_dir / 'latency_benchmark.json'
        if latency_path.exists():
            with open(latency_path) as f:
                data["latency"] = json.load(f)

        all_results.append(data)

    return all_results


def create_accuracy_table_latex(
    results_dirs: list[Path],
    dataset_names: list[str],
    save_path: Path,
):
    """Create LaTeX table comparing accuracy across all datasets."""
    save_path.parent.mkdir(exist_ok=True)

    all_results = load_all_dataset_results(results_dirs, dataset_names)

    latex = r"""\begin{table}[t]
\centering
\caption{Model accuracy across datasets. Count queries report MAE and Pearson correlation; Distance queries report MAE and percentage within $\pm$1 hop.}
\label{tab:accuracy_comparison}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
& \multicolumn{2}{c}{\textbf{Graph}} & \multicolumn{3}{c}{\textbf{Count Queries}} & \multicolumn{2}{c}{\textbf{Distance Queries}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-8}
\textbf{Dataset} & Nodes & Edges & MAE & MAPE & $r$ & MAE & Acc$_{\pm1}$ \\
\midrule
"""

    for data in all_results:
        name = data["name"]
        acc = data.get("eval", {}).get("accuracy", {})
        latency = data.get("latency", {}).get("config", {})

        nodes = latency.get("graph_nodes", 0)
        edges = latency.get("graph_edges", 0)

        count_mae = acc.get("count_mae", 0)
        count_mape = acc.get("count_mape", 0)
        count_r = acc.get("count_pearson_r", 0)
        dist_mae = acc.get("dist_mae", 0)
        dist_acc = acc.get("dist_accuracy_1hop", 0)

        latex += f"{name} & {nodes:,} & {edges:,} & {count_mae:.2f} & {count_mape:.1f}\\% & {count_r:.3f} & {dist_mae:.2f} & {dist_acc:.1f}\\% \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}"""

    with open(save_path, 'w') as f:
        f.write(latex)

    print(f"Saved accuracy table to {save_path}")
    return latex


def create_latency_table_latex(
    results_dirs: list[Path],
    dataset_names: list[str],
    save_path: Path,
):
    """Create LaTeX table comparing latency across all datasets."""
    save_path.parent.mkdir(exist_ok=True)

    all_results = load_all_dataset_results(results_dirs, dataset_names)

    # Check if Kuzu results are available
    has_kuzu = any("kuzu" in data.get("latency", {}) and "error" not in data.get("latency", {}).get("kuzu", {})
                   for data in all_results)

    if has_kuzu:
        latex = r"""\begin{table}[t]
\centering
\caption{Latency comparison: Neural network model vs NetworkX (in-memory) vs Kuzu (embedded database). All times in milliseconds.}
\label{tab:latency_comparison}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{2}{c}{\textbf{NN Model}} & \textbf{NetworkX} & \textbf{Kuzu} & \multicolumn{2}{c}{\textbf{Speedup}} \\
\cmidrule(lr){2-3} \cmidrule(lr){6-7}
\textbf{Dataset} & Mean & P99 & Mean & Mean & vs NX & vs Kuzu \\
\midrule
"""
    else:
        latex = r"""\begin{table}[t]
\centering
\caption{Latency comparison: Neural network model vs NetworkX (in-memory graph library). All times in milliseconds.}
\label{tab:latency_comparison}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
& \multicolumn{2}{c}{\textbf{NN Model}} & \multicolumn{2}{c}{\textbf{NetworkX}} & \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
\textbf{Dataset} & Mean & P99 & Mean & P99 & \textbf{Speedup} \\
\midrule
"""

    for data in all_results:
        name = data["name"]
        latency = data.get("latency", {})
        overall = latency.get("overall", {})

        model_mean = overall.get("model", {}).get("total", {}).get("mean_ms", 0)
        model_p99 = overall.get("model", {}).get("total", {}).get("p99_ms", 0)
        gt_mean = overall.get("ground_truth", {}).get("mean_ms", 0)
        gt_p99 = overall.get("ground_truth", {}).get("p99_ms", 0)
        speedup = overall.get("speedup_mean", 0)

        if has_kuzu:
            kuzu = latency.get("kuzu", {})
            kuzu_mean = kuzu.get("mean_ms", 0) if "error" not in kuzu else 0
            kuzu_speedup = kuzu.get("speedup_vs_model", 0) if "error" not in kuzu else 0

            latex += f"{name} & {model_mean:.3f} & {model_p99:.3f} & {gt_mean:.3f} & {kuzu_mean:.3f} & {speedup:.1f}$\\times$ & {kuzu_speedup:.1f}$\\times$ \\\\\n"
        else:
            latex += f"{name} & {model_mean:.3f} & {model_p99:.3f} & {gt_mean:.3f} & {gt_p99:.3f} & {speedup:.1f}$\\times$ \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}"""

    with open(save_path, 'w') as f:
        f.write(latex)

    print(f"Saved latency table to {save_path}")
    return latex


def create_throughput_table_latex(
    results_dirs: list[Path],
    dataset_names: list[str],
    save_path: Path,
):
    """Create LaTeX table comparing batch throughput across all datasets."""
    save_path.parent.mkdir(exist_ok=True)

    all_results = load_all_dataset_results(results_dirs, dataset_names)

    latex = r"""\begin{table}[t]
\centering
\caption{Batch inference throughput (queries per second) at different batch sizes.}
\label{tab:throughput_comparison}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Dataset} & \textbf{BS=1} & \textbf{BS=32} & \textbf{BS=128} & \textbf{BS=512} & \textbf{Max} \\
\midrule
"""

    for data in all_results:
        name = data["name"]
        batch = data.get("latency", {}).get("batch_throughput", {})

        bs1 = batch.get(1, batch.get("1", {})).get("throughput_qps", 0)
        bs32 = batch.get(32, batch.get("32", {})).get("throughput_qps", 0)
        bs128 = batch.get(128, batch.get("128", {})).get("throughput_qps", 0)
        bs512 = batch.get(512, batch.get("512", {})).get("throughput_qps", 0)

        max_tp = max(bs1, bs32, bs128, bs512)

        latex += f"{name} & {bs1:,.0f} & {bs32:,.0f} & {bs128:,.0f} & {bs512:,.0f} & {max_tp:,.0f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}"""

    with open(save_path, 'w') as f:
        f.write(latex)

    print(f"Saved throughput table to {save_path}")
    return latex


def plot_multi_dataset_comparison(
    results_dirs: list[Path],
    dataset_names: list[str],
    save_path: Path,
):
    save_path.parent.mkdir(exist_ok=True)

    all_results = load_all_dataset_results(results_dirs, dataset_names)

    count_maes = []
    dist_maes = []
    speedups = []
    kuzu_speedups = []
    has_kuzu = False

    for data in all_results:
        acc = data.get("eval", {}).get("accuracy", {})
        latency = data.get("latency", {})
        overall = latency.get("overall", {})
        kuzu = latency.get("kuzu", {})

        count_maes.append(acc.get('count_mae', 0))
        dist_maes.append(acc.get('dist_mae', 0))
        speedups.append(overall.get('speedup_mean', 0))

        if kuzu and "error" not in kuzu:
            has_kuzu = True
            kuzu_speedups.append(kuzu.get('speedup_vs_model', 0))
        else:
            kuzu_speedups.append(0)

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.5))
    x = np.arange(len(dataset_names))
    width = 0.6

    ax1 = axes[0]
    ax1.bar(x, count_maes, width, color=COLORS['primary'], edgecolor='white', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax1.set_ylabel('MAE')
    ax1.set_title('(a) Count Query MAE')

    ax2 = axes[1]
    ax2.bar(x, dist_maes, width, color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax2.set_ylabel('MAE')
    ax2.set_title('(b) Distance Query MAE')

    ax3 = axes[2]
    if has_kuzu:
        width = 0.35
        ax3.bar(x - width/2, speedups, width, color=COLORS['tertiary'], edgecolor='white', linewidth=0.5, label='vs NetworkX')
        ax3.bar(x + width/2, kuzu_speedups, width, color=COLORS['quaternary'], edgecolor='white', linewidth=0.5, label='vs Kuzu')
        ax3.legend(loc='upper right', fontsize=7)
    else:
        ax3.bar(x, speedups, width, color=COLORS['tertiary'], edgecolor='white', linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax3.set_ylabel('Speedup')
    ax3.set_title('(c) Inference Speedup')
    ax3.axhline(1, color=COLORS['gray'], linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300)
    plt.close()

    print(f"Saved multi-dataset comparison to {save_path}")

    # Also create LaTeX tables
    tables_dir = save_path.parent
    create_accuracy_table_latex(results_dirs, dataset_names, tables_dir / 'table_accuracy.tex')
    create_latency_table_latex(results_dirs, dataset_names, tables_dir / 'table_latency.tex')
    create_throughput_table_latex(results_dirs, dataset_names, tables_dir / 'table_throughput.tex')


def create_all_visualizations(output_dir: Path, figures_dir: Path, dataset_name: str = ""):
    figures_dir.mkdir(exist_ok=True)

    print(f"\nGenerating visualizations for {output_dir}...")

    eval_results, history, queries = load_results(output_dir)

    predictions, ground_truths, query_types = generate_predictions(output_dir)

    prefix = f"{dataset_name}_" if dataset_name else ""

    if eval_results:
        create_metrics_table(eval_results, figures_dir / f'{prefix}metrics_table.tex', dataset_name or "Test Set")

    if history:
        plot_training_curves(history, figures_dir / f'{prefix}training_curves.pdf')

    if predictions is not None:
        plot_predictions_scatter(predictions, ground_truths, query_types,
                                 figures_dir / f'{prefix}predictions_scatter.pdf')

        plot_error_distribution(predictions, ground_truths, query_types,
                                figures_dir / f'{prefix}error_distribution.pdf')

    print(f"\nAll visualizations saved to {figures_dir}/")


def plot_latency_comparison(
    latency_results: dict,
    save_path: Path,
):
    """
    Create latency comparison visualization showing NN model vs ground truth.

    Args:
        latency_results: Results from benchmark_latency.py
        save_path: Path to save the figure
    """
    save_path.parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.5))

    # Get raw timing data
    model_times = np.array(latency_results.get("raw_times", {}).get("model_total_ms", []))
    gt_times = np.array(latency_results.get("raw_times", {}).get("ground_truth_ms", []))
    encoding_times = np.array(latency_results.get("raw_times", {}).get("model_encoding_ms", []))
    inference_times = np.array(latency_results.get("raw_times", {}).get("model_inference_ms", []))

    if len(model_times) == 0 or len(gt_times) == 0:
        print("No raw timing data available for latency visualization")
        plt.close()
        return

    # (a) Latency distribution histogram
    ax1 = axes[0, 0]
    bins = np.linspace(0, max(np.percentile(gt_times, 99), np.percentile(model_times, 99)), 50)
    ax1.hist(model_times, bins=bins, alpha=0.7, label='NN Model', color=COLORS['primary'], edgecolor='white', linewidth=0.3)
    ax1.hist(gt_times, bins=bins, alpha=0.7, label='Ground Truth', color=COLORS['secondary'], edgecolor='white', linewidth=0.3)
    ax1.axvline(np.mean(model_times), color=COLORS['primary'], linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.axvline(np.mean(gt_times), color=COLORS['secondary'], linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Latency (ms)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(a) Latency Distribution')
    ax1.legend(loc='upper right')

    # (b) Box plot comparison
    ax2 = axes[0, 1]
    bp = ax2.boxplot(
        [model_times, gt_times],
        tick_labels=['NN Model', 'Ground Truth'],
        patch_artist=True,
        widths=0.6,
    )
    bp['boxes'][0].set_facecolor(COLORS['light_blue'])
    bp['boxes'][1].set_facecolor(COLORS['light_red'])
    for box in bp['boxes']:
        box.set_edgecolor(COLORS['gray'])
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('(b) Latency Box Plot')

    # Add speedup annotation
    speedup = np.mean(gt_times) / np.mean(model_times)
    ax2.text(0.5, 0.95, f'Speedup: {speedup:.1f}x',
             transform=ax2.transAxes, ha='center', va='top',
             fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLORS['light_gray']))

    # (c) Model latency breakdown (stacked bar)
    ax3 = axes[1, 0]
    if len(encoding_times) > 0 and len(inference_times) > 0:
        mean_encoding = np.mean(encoding_times)
        mean_inference = np.mean(inference_times)
        mean_gt = np.mean(gt_times)

        x = np.array([0, 1])
        width = 0.5

        # NN Model stacked bar
        ax3.bar(0, mean_encoding, width, label='Encoding', color=COLORS['light_blue'], edgecolor='white')
        ax3.bar(0, mean_inference, width, bottom=mean_encoding, label='Inference', color=COLORS['primary'], edgecolor='white')

        # Ground truth bar
        ax3.bar(1, mean_gt, width, label='Query Execution', color=COLORS['secondary'], edgecolor='white')

        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['NN Model', 'Ground Truth'])
        ax3.set_ylabel('Mean Latency (ms)')
        ax3.set_title('(c) Latency Breakdown')
        ax3.legend(loc='upper right')

        # Add value labels
        ax3.text(0, mean_encoding + mean_inference + 0.01 * mean_gt, f'{mean_encoding + mean_inference:.3f}ms',
                 ha='center', va='bottom', fontsize=7)
        ax3.text(1, mean_gt + 0.01 * mean_gt, f'{mean_gt:.3f}ms',
                 ha='center', va='bottom', fontsize=7)

    # (d) Percentile comparison
    ax4 = axes[1, 1]
    percentiles = [50, 90, 95, 99]
    model_pcts = [np.percentile(model_times, p) for p in percentiles]
    gt_pcts = [np.percentile(gt_times, p) for p in percentiles]

    x = np.arange(len(percentiles))
    width = 0.35

    ax4.bar(x - width/2, model_pcts, width, label='NN Model', color=COLORS['primary'], edgecolor='white')
    ax4.bar(x + width/2, gt_pcts, width, label='Ground Truth', color=COLORS['secondary'], edgecolor='white')

    ax4.set_xticks(x)
    ax4.set_xticklabels([f'P{p}' for p in percentiles])
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title('(d) Latency Percentiles')
    ax4.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300)
    plt.close()

    print(f"Saved latency comparison to {save_path}")


def plot_latency_by_complexity(
    latency_results: dict,
    save_path: Path,
):
    """
    Plot latency scaling with query complexity.

    Args:
        latency_results: Results from benchmark_latency.py
        save_path: Path to save the figure
    """
    save_path.parent.mkdir(exist_ok=True)

    by_complexity = latency_results.get("by_complexity", {})
    count_by_radius = by_complexity.get("count_by_radius", {})
    dist_by_hops = by_complexity.get("distance_by_max_hops", {})

    if not count_by_radius and not dist_by_hops:
        print("No complexity data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))

    # (a) COUNT queries by radius
    ax1 = axes[0]
    if count_by_radius:
        radii = sorted(count_by_radius.keys())
        model_times = [count_by_radius[r]["model_mean_ms"] for r in radii]
        gt_times = [count_by_radius[r]["gt_mean_ms"] for r in radii]

        x = np.arange(len(radii))
        width = 0.35

        ax1.bar(x - width/2, model_times, width, label='NN Model', color=COLORS['primary'], edgecolor='white')
        ax1.bar(x + width/2, gt_times, width, label='Ground Truth', color=COLORS['secondary'], edgecolor='white')

        ax1.set_xticks(x)
        ax1.set_xticklabels([str(r) for r in radii])
        ax1.set_xlabel('Radius (k-hops)')
        ax1.set_ylabel('Mean Latency (ms)')
        ax1.set_title('(a) COUNT Query Latency by Radius')
        ax1.legend(loc='upper left')

    # (b) DISTANCE queries by max_hops
    ax2 = axes[1]
    if dist_by_hops:
        hops = sorted(dist_by_hops.keys())
        model_times = [dist_by_hops[h]["model_mean_ms"] for h in hops]
        gt_times = [dist_by_hops[h]["gt_mean_ms"] for h in hops]

        x = np.arange(len(hops))
        width = 0.35

        ax2.bar(x - width/2, model_times, width, label='NN Model', color=COLORS['primary'], edgecolor='white')
        ax2.bar(x + width/2, gt_times, width, label='Ground Truth', color=COLORS['secondary'], edgecolor='white')

        ax2.set_xticks(x)
        ax2.set_xticklabels([str(h) for h in hops])
        ax2.set_xlabel('Max Hops')
        ax2.set_ylabel('Mean Latency (ms)')
        ax2.set_title('(b) DISTANCE Query Latency by Max Hops')
        ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300)
    plt.close()

    print(f"Saved latency by complexity to {save_path}")


def plot_batch_throughput(
    latency_results: dict,
    save_path: Path,
):
    """
    Plot batch throughput analysis.

    Args:
        latency_results: Results from benchmark_latency.py
        save_path: Path to save the figure
    """
    save_path.parent.mkdir(exist_ok=True)

    batch_results = latency_results.get("batch_throughput", {})
    if not batch_results:
        print("No batch throughput data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.5))

    batch_sizes = sorted(batch_results.keys())
    throughputs = [batch_results[bs]["throughput_qps"] for bs in batch_sizes]
    latencies = [batch_results[bs]["latency_per_query_ms"] for bs in batch_sizes]

    # (a) Throughput vs batch size
    ax1 = axes[0]
    ax1.plot(batch_sizes, throughputs, 'o-', color=COLORS['primary'], linewidth=1.5, markersize=6)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (queries/sec)')
    ax1.set_title('(a) Throughput vs Batch Size')
    ax1.set_xscale('log', base=2)

    # Add annotation for max throughput
    max_idx = np.argmax(throughputs)
    ax1.annotate(f'{throughputs[max_idx]:,.0f} q/s',
                 xy=(batch_sizes[max_idx], throughputs[max_idx]),
                 xytext=(batch_sizes[max_idx], throughputs[max_idx] * 1.1),
                 ha='center', fontsize=7)

    # (b) Latency per query vs batch size
    ax2 = axes[1]
    ax2.plot(batch_sizes, latencies, 'o-', color=COLORS['secondary'], linewidth=1.5, markersize=6)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Latency per Query (ms)')
    ax2.set_title('(b) Amortized Latency vs Batch Size')
    ax2.set_xscale('log', base=2)

    # Add ground truth comparison line if available
    gt_mean = latency_results.get("overall", {}).get("ground_truth", {}).get("mean_ms", 0)
    if gt_mean > 0:
        ax2.axhline(gt_mean, color=COLORS['gray'], linestyle='--', linewidth=1.0, alpha=0.7)
        ax2.text(batch_sizes[-1], gt_mean * 1.05, f'Ground Truth: {gt_mean:.3f}ms',
                 ha='right', va='bottom', fontsize=7, color=COLORS['gray'])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300)
    plt.close()

    print(f"Saved batch throughput to {save_path}")


def plot_speedup_summary(
    latency_results: dict,
    save_path: Path,
):
    """
    Create a summary speedup visualization for publication.

    Args:
        latency_results: Results from benchmark_latency.py
        save_path: Path to save the figure
    """
    save_path.parent.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))

    # Collect speedup data
    categories = []
    speedups = []
    colors = []

    # Overall speedup
    overall = latency_results.get("overall", {})
    if overall:
        categories.append("Overall")
        speedups.append(overall.get("speedup_mean", 0))
        colors.append(COLORS['primary'])

    # By query type
    by_type = latency_results.get("by_query_type", {})
    if "count" in by_type:
        categories.append("COUNT")
        speedups.append(by_type["count"].get("speedup_mean", 0))
        colors.append(COLORS['light_blue'])
    if "distance" in by_type:
        categories.append("DISTANCE")
        speedups.append(by_type["distance"].get("speedup_mean", 0))
        colors.append(COLORS['light_red'])

    if not categories:
        print("No speedup data available")
        plt.close()
        return

    x = np.arange(len(categories))
    bars = ax.bar(x, speedups, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for bar, speedup in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.axhline(1, color=COLORS['gray'], linestyle='--', linewidth=0.75, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Neural Network Speedup over Graph Queries')
    ax.set_ylim(0, max(speedups) * 1.2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(str(save_path).replace('.pdf', '.png'), dpi=300)
    plt.close()

    print(f"Saved speedup summary to {save_path}")


def create_latency_visualizations(output_dir: Path, figures_dir: Path, dataset_name: str = ""):
    """Generate all latency-related visualizations."""
    figures_dir.mkdir(exist_ok=True)

    latency_path = output_dir / "latency_benchmark.json"
    if not latency_path.exists():
        print(f"No latency benchmark results found at {latency_path}")
        print("Run: python -m scripts.benchmark_latency --model-dir <model_dir>")
        return

    with open(latency_path) as f:
        latency_results = json.load(f)

    prefix = f"{dataset_name}_" if dataset_name else ""

    plot_latency_comparison(latency_results, figures_dir / f'{prefix}latency_comparison.pdf')
    plot_latency_by_complexity(latency_results, figures_dir / f'{prefix}latency_complexity.pdf')
    plot_batch_throughput(latency_results, figures_dir / f'{prefix}batch_throughput.pdf')
    plot_speedup_summary(latency_results, figures_dir / f'{prefix}speedup_summary.pdf')

    print(f"\nLatency visualizations saved to {figures_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate publication visualizations")
    parser.add_argument("--output-dir", type=str, default="output", help="Model output directory")
    parser.add_argument("--figures-dir", type=str, default="figures", help="Figures output directory")
    parser.add_argument("--dataset-name", type=str, default="", help="Dataset name for labeling")
    parser.add_argument("--compare", type=str, nargs="+", help="Compare multiple output directories")
    parser.add_argument("--compare-names", type=str, nargs="+", help="Names for comparison datasets")
    parser.add_argument("--latency", action="store_true", help="Generate latency visualizations")
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)

    if args.compare:
        result_dirs = [Path(d) for d in args.compare]
        names = args.compare_names if args.compare_names else [d.name for d in result_dirs]
        plot_multi_dataset_comparison(result_dirs, names, figures_dir / 'comparison.pdf')

        for result_dir, name in zip(result_dirs, names):
            create_all_visualizations(result_dir, figures_dir, name)
            if args.latency:
                create_latency_visualizations(result_dir, figures_dir, name)
    else:
        create_all_visualizations(Path(args.output_dir), figures_dir, args.dataset_name)
        if args.latency:
            create_latency_visualizations(Path(args.output_dir), figures_dir, args.dataset_name)


if __name__ == '__main__':
    main()
