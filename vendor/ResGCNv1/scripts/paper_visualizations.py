"""
Paper Visualizations for ResGCN Infant Gait Age Estimation

Generates publication-quality figures from extract() output (.npz files):
1. Scatter plot: predicted vs true age
2. Bland-Altman plot: agreement analysis
3. Age-group error boxplot: per-group MAE distribution
4. LDL distribution visualization: predicted vs target distributions
5. UMAP feature embedding: FC-layer features colored by age

Usage:
    python scripts/paper_visualizations.py --npz <path_to_npz> --output_dir <output_dir>
    python scripts/paper_visualizations.py --npz extraction_resgcn_coco.npz --output_dir paper_figures/
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats


# ========== Plot Style Configuration ==========
STYLE = {
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}
plt.rcParams.update(STYLE)


def load_data(npz_path):
    """Load extraction data from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    result = {}
    for key in data.files:
        result[key] = data[key]
    return result


# ========== 1. Scatter Plot ==========
def plot_scatter(npz_path, output_dir, figsize=(7, 7)):
    """Predicted vs True age scatter plot with regression line.
    
    Includes: identity line (y=x), linear regression, R², MAE annotation.
    """
    data = load_data(npz_path)
    pred = data['pred_expectations']
    true = data['target_expectations']

    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(true, pred)
    spearman_r, _ = stats.spearmanr(true, pred)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter points with transparency
    ax.scatter(true, pred, alpha=0.5, s=40, c='#4C72B0', edgecolors='white', linewidth=0.5, zorder=3)

    # Identity line (y=x)
    lims = [min(true.min(), pred.min()) - 1, max(true.max(), pred.max()) + 1]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Ideal (y=x)', zorder=1)

    # Regression line
    x_fit = np.linspace(lims[0], lims[1], 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', linewidth=1.5, label=f'Fit (y={slope:.2f}x+{intercept:.2f})', zorder=2)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True Age (months)')
    ax.set_ylabel('Predicted Age (months)')
    ax.set_title('Predicted vs True Age')
    ax.set_aspect('equal')

    # Annotation box
    textstr = f'MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR² = {r2:.3f}\nSpearman ρ = {spearman_r:.3f}\nn = {len(true)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top', bbox=props)

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, 'scatter_pred_vs_true.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'Saved scatter plot: {save_path}')


# ========== 2. Bland-Altman Plot ==========
def plot_bland_altman(npz_path, output_dir, figsize=(8, 6)):
    """Bland-Altman plot for agreement analysis.
    
    X-axis: mean of (pred, true)
    Y-axis: difference (pred - true)
    Shows: mean bias, ±1.96 SD limits of agreement
    """
    data = load_data(npz_path)
    pred = data['pred_expectations']
    true = data['target_expectations']

    mean_vals = (pred + true) / 2.0
    diff_vals = pred - true

    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals, ddof=1)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(mean_vals, diff_vals, alpha=0.5, s=40, c='#4C72B0', edgecolors='white', linewidth=0.5)

    # Mean bias line
    ax.axhline(y=mean_diff, color='red', linestyle='-', linewidth=1.5,
               label=f'Mean bias = {mean_diff:.2f}')
    # Upper LOA
    ax.axhline(y=upper_loa, color='gray', linestyle='--', linewidth=1,
               label=f'+1.96 SD = {upper_loa:.2f}')
    # Lower LOA
    ax.axhline(y=lower_loa, color='gray', linestyle='--', linewidth=1,
               label=f'−1.96 SD = {lower_loa:.2f}')
    # Zero line
    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.5, alpha=0.5)

    # Fill LOA region
    ax.fill_between(ax.get_xlim(), lower_loa, upper_loa, alpha=0.08, color='gray')

    ax.set_xlabel('Mean of Predicted and True Age (months)')
    ax.set_ylabel('Predicted − True Age (months)')
    ax.set_title('Bland-Altman Plot')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, 'bland_altman.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'Saved Bland-Altman plot: {save_path}')


# ========== 3. Age-Group Error Boxplot ==========
def plot_age_group_error(npz_path, output_dir, groups=None, figsize=(9, 6)):
    """Per-age-group MAE boxplot.
    
    Args:
        groups: list of (min, max) tuples for age groups.
                Default: [(12,16), (16,20), (20,24), (24,28), (28,32), (32,36)]
    """
    data = load_data(npz_path)
    pred = data['pred_expectations']
    true = data['target_expectations']

    if groups is None:
        groups = [(12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36)]

    errors_per_group = []
    labels = []
    counts = []

    for g_min, g_max in groups:
        mask = (true >= g_min) & (true < g_max)
        if mask.sum() == 0:
            continue
        group_errors = np.abs(pred[mask] - true[mask])
        errors_per_group.append(group_errors)
        labels.append(f'{g_min}-{g_max}')
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(errors_per_group, patch_artist=True, labels=labels,
                    medianprops=dict(color='red', linewidth=1.5),
                    whiskerprops=dict(linewidth=1),
                    boxprops=dict(linewidth=1))

    # Color boxes with a gradient
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(errors_per_group)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add sample count annotations
    for i, (count, label) in enumerate(zip(counts, labels)):
        ax.text(i + 1, ax.get_ylim()[1] * 0.92, f'n={count}',
                ha='center', va='top', fontsize=9, color='gray')

    # Add mean MAE per group as markers
    for i, errs in enumerate(errors_per_group):
        ax.scatter(i + 1, np.mean(errs), marker='D', color='darkred', s=30, zorder=5)

    ax.set_xlabel('Age Group (months)')
    ax.set_ylabel('Absolute Error (months)')
    ax.set_title('Prediction Error by Age Group')
    ax.grid(True, alpha=0.3, axis='y')

    # Add overall MAE annotation
    overall_mae = np.mean(np.abs(pred - true))
    ax.axhline(y=overall_mae, color='orange', linestyle='--', linewidth=1,
               label=f'Overall MAE = {overall_mae:.2f}', alpha=0.7)
    ax.legend(loc='upper left')

    save_path = os.path.join(output_dir, 'age_group_error.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'Saved age-group error plot: {save_path}')


# ========== 4. LDL Distribution Visualization ==========
def plot_ldl_distribution(npz_path, output_dir, sample_indices=None, figsize=(12, 4)):
    """Visualize predicted vs target label distributions for selected samples.
    
    Args:
        sample_indices: list of sample indices to visualize. Default: auto-select 3 samples
                        (one good, one medium, one poor prediction).
    """
    data = load_data(npz_path)
    pred_out = data['out']              # softmax output: (N, num_class)
    target_dist = data['label']         # target distribution: (N, num_class)
    bin_centers = data['bin_centers']    # bin centers: (num_class,)
    pred_exp = data['pred_expectations']
    true_exp = data['target_expectations']

    # Auto-select samples if not specified
    if sample_indices is None:
        abs_errors = np.abs(pred_exp - true_exp)
        sorted_idx = np.argsort(abs_errors)
        n = len(sorted_idx)
        # Best, median, worst prediction
        sample_indices = [
            sorted_idx[0],          # best
            sorted_idx[n // 2],     # median
            sorted_idx[-1],         # worst
        ]
        titles = ['Best Prediction', 'Median Prediction', 'Worst Prediction']
    else:
        titles = [f'Sample {i}' for i in sample_indices]

    num_samples = len(sample_indices)
    fig, axes = plt.subplots(1, num_samples, figsize=(figsize[0], figsize[1]))
    if num_samples == 1:
        axes = [axes]

    bar_width = (bin_centers[1] - bin_centers[0]) * 0.35 if len(bin_centers) > 1 else 1.0

    for ax, idx, title in zip(axes, sample_indices, titles):
        pred_dist = pred_out[idx]
        tgt_dist = target_dist[idx]

        x_pred = bin_centers - bar_width / 2
        x_tgt = bin_centers + bar_width / 2

        ax.bar(x_tgt, tgt_dist, width=bar_width, alpha=0.7, color='#55A868',
               label='Target', edgecolor='white', linewidth=0.5)
        ax.bar(x_pred, pred_dist, width=bar_width, alpha=0.7, color='#4C72B0',
               label='Predicted', edgecolor='white', linewidth=0.5)

        # Mark expectations
        p_exp = pred_exp[idx]
        t_exp = true_exp[idx]
        ax.axvline(x=p_exp, color='#4C72B0', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(x=t_exp, color='#55A868', linestyle='--', linewidth=1.5, alpha=0.8)

        error = abs(p_exp - t_exp)
        ax.set_title(f'{title}\nTrue={t_exp:.1f}, Pred={p_exp:.1f}, |Err|={error:.1f}',
                     fontsize=11)
        ax.set_xlabel('Age (months)')
        ax.set_ylabel('Probability')
        ax.set_xticks(bin_centers)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ldl_distributions.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'Saved LDL distribution plot: {save_path}')


# ========== 5. Summary Table (print to console) ==========
def print_summary(npz_path):
    """Print a summary table of all metrics."""
    data = load_data(npz_path)
    pred = data['pred_expectations']
    true = data['target_expectations']

    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
    spearman_r, spearman_p = stats.spearmanr(true, pred)
    kendall_t, kendall_p = stats.kendalltau(true, pred)

    print('\n' + '=' * 50)
    print('  Evaluation Metrics Summary')
    print('=' * 50)
    print(f'  Samples:      {len(true)}')
    print(f'  MAE:          {mae:.4f} months')
    print(f'  MSE:          {mse:.4f}')
    print(f'  RMSE:         {rmse:.4f} months')
    print(f'  R²:           {r2:.4f}')
    print(f'  Spearman ρ:   {spearman_r:.4f} (p={spearman_p:.2e})')
    print(f'  Kendall τ:    {kendall_t:.4f} (p={kendall_p:.2e})')
    print('=' * 50 + '\n')


# ========== 6. UMAP Feature Embedding ==========
def plot_umap(npz_path, output_dir, figsize=(8, 7)):
    """UMAP visualization of FC-layer-preceding features colored by true age.
    
    The 'feature' key in the npz file contains the 256-dim feature vector
    from global pooling output, before dropout and FC layer.
    """
    try:
        import umap
    except ImportError:
        print('ERROR: umap-learn not installed. Run: pip install umap-learn')
        return

    data = load_data(npz_path)
    features = data['feature']            # (N, feature_dim), e.g., (51, 256)
    true_ages = data['target_expectations']
    pred_ages = data['pred_expectations']

    # Flatten features if needed (e.g., if shape is (N, dim, 1, ...))
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)

    print(f'  Feature shape: {features.shape}')
    print(f'  Running UMAP reduction...')

    # UMAP reduction to 2D
    n_neighbors = min(15, len(features) - 1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.3,
        n_components=2,
        metric='euclidean',
        random_state=42
    )
    embedding = reducer.fit_transform(features)

    # --- Plot 1: colored by TRUE age ---
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=true_ages, cmap='RdYlBu_r', s=60,
        edgecolors='white', linewidth=0.5, alpha=0.85
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('True Age (months)', fontsize=12)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_title('UMAP Feature Embedding (colored by True Age)')
    ax.grid(True, alpha=0.2)

    save_path = os.path.join(output_dir, 'umap_true_age.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'Saved UMAP (true age): {save_path}')

    # --- Plot 2: colored by PREDICTION ERROR ---
    abs_errors = np.abs(pred_ages - true_ages)
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=abs_errors, cmap='YlOrRd', s=60,
        edgecolors='white', linewidth=0.5, alpha=0.85
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('|Prediction Error| (months)', fontsize=12)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_title('UMAP Feature Embedding (colored by Prediction Error)')
    ax.grid(True, alpha=0.2)

    save_path = os.path.join(output_dir, 'umap_pred_error.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'Saved UMAP (prediction error): {save_path}')


# ========== CLI Entry Point ==========
def main():
    parser = argparse.ArgumentParser(description='Generate paper visualizations from extract() output')
    parser.add_argument('--npz', type=str, required=True, help='Path to extraction .npz file')
    parser.add_argument('--output_dir', type=str, default='paper_figures/', help='Output directory for figures')
    parser.add_argument('--plots', type=str, nargs='+',
                        default=['scatter', 'bland_altman', 'age_group', 'ldl', 'umap', 'summary'],
                        help='Which plots to generate')
    parser.add_argument('--samples', type=int, nargs='+', default=None,
                        help='Sample indices for LDL distribution plot')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_funcs = {
        'scatter': lambda: plot_scatter(args.npz, args.output_dir),
        'bland_altman': lambda: plot_bland_altman(args.npz, args.output_dir),
        'age_group': lambda: plot_age_group_error(args.npz, args.output_dir),
        'ldl': lambda: plot_ldl_distribution(args.npz, args.output_dir, sample_indices=args.samples),
        'umap': lambda: plot_umap(args.npz, args.output_dir),
        'summary': lambda: print_summary(args.npz),
    }

    for plot_name in args.plots:
        if plot_name in plot_funcs:
            print(f'\nGenerating: {plot_name}...')
            plot_funcs[plot_name]()
        else:
            print(f'Unknown plot type: {plot_name}. Available: {list(plot_funcs.keys())}')

    print(f'\nAll figures saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
