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
def get_aggregated_data(npz_path, agg_by_case=False):
    data = load_data(npz_path)
    pred = data['pred_expectations']
    true = data['target_expectations']
    
    if agg_by_case and 'name' in data:
        names = data['name']
        case_dict = {}
        for i, name in enumerate(names):
            parts = str(name).split('_')
            case_id = '_'.join(parts[:2]) if len(parts) >= 2 else str(name)
            if case_id not in case_dict:
                case_dict[case_id] = {'pred': [], 'true': []}
            case_dict[case_id]['pred'].append(pred[i])
            case_dict[case_id]['true'].append(true[i])
            
        case_true = []
        case_pred = []
        for case_id, values in case_dict.items():
            case_true.append(np.mean(values['true']))
            case_pred.append(np.mean(values['pred']))
        return np.array(case_pred), np.array(case_true)
    return pred, true

def plot_scatter(npz_path, output_dir, figsize=(7, 7), npz_retard_path=None, agg_by_case=False):
    """Predicted vs True age scatter plot with regression line.
    
    Includes: identity line (y=x), linear regression, R², MAE annotation.
    """
    pred_val, true_val = get_aggregated_data(npz_path, agg_by_case)

    mae_val = np.mean(np.abs(pred_val - true_val))
    rmse_val = np.sqrt(np.mean((pred_val - true_val) ** 2))
    r2_val = 1 - np.sum((true_val - pred_val) ** 2) / np.sum((true_val - np.mean(true_val)) ** 2)
    slope_val, intercept_val, r_value_val, p_value_val, std_err_val = stats.linregress(true_val, pred_val)
    spearman_r_val, _ = stats.spearmanr(true_val, pred_val)

    fig, ax = plt.subplots(figsize=figsize)

    # Validation data (Blue)
    ax.scatter(true_val, pred_val, alpha=0.5, s=40, c='#4C72B0', edgecolors='white', linewidth=0.5, zorder=3, label='Validation')

    # Regression line
    x_fit_val = np.linspace(true_val.min(), true_val.max(), 100)
    y_fit_val = slope_val * x_fit_val + intercept_val
    ax.plot(x_fit_val, y_fit_val, color='#4C72B0', linestyle='-', linewidth=1.5, label=f'Val Fit (y={slope_val:.2f}x+{intercept_val:.2f})', zorder=2)

    all_true = list(true_val)
    all_pred = list(pred_val)

    textstr = f'[Validation]\nMAE = {mae_val:.2f}\nRMSE = {rmse_val:.2f}\nR² = {r2_val:.3f}'

    if npz_retard_path:
        pred_retard, true_retard = get_aggregated_data(npz_retard_path, agg_by_case)

        mae_retard = np.mean(np.abs(pred_retard - true_retard))
        rmse_retard = np.sqrt(np.mean((pred_retard - true_retard) ** 2))
        r2_retard = 1 - np.sum((true_retard - pred_retard) ** 2) / np.sum((true_retard - np.mean(true_retard)) ** 2)
        slope_retard, intercept_retard, _, _, _ = stats.linregress(true_retard, pred_retard)

        # Retard data (Red)
        ax.scatter(true_retard, pred_retard, alpha=0.5, s=40, c='#C44E52', edgecolors='white', linewidth=0.5, zorder=4, label='Retard')

        x_fit_retard = np.linspace(true_retard.min(), true_retard.max(), 100)
        y_fit_retard = slope_retard * x_fit_retard + intercept_retard
        ax.plot(x_fit_retard, y_fit_retard, color='#C44E52', linestyle='-', linewidth=1.5, label=f'Retard Fit (y={slope_retard:.2f}x+{intercept_retard:.2f})', zorder=2)

        all_true.extend(true_retard)
        all_pred.extend(pred_retard)

        textstr += f'\n\n[Retard]\nMAE = {mae_retard:.2f}\nRMSE = {rmse_retard:.2f}\nR² = {r2_retard:.3f}'
    else:
        textstr += f'\nSpearman ρ = {spearman_r_val:.3f}\nn = {len(true_val)}'

    # Identity line (y=x)
    lims = [min(min(all_true), min(all_pred)) - 1, max(max(all_true), max(all_pred)) + 1]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Ideal (y=x)', zorder=1)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True Age (months)')
    ax.set_ylabel('Predicted Age (months)')
    title = 'Predicted vs True Age' + (' (Averaged by Case ID)' if agg_by_case else '')
    ax.set_title(title)
    ax.set_aspect('equal')

    # Annotation box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top', bbox=props)

    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    filename = 'scatter_pred_vs_true_by_case.png' if agg_by_case else 'scatter_pred_vs_true.png'
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path)
    plt.close(fig)
    print(f'Saved scatter plot: {save_path}')


# ========== 2. Bland-Altman Plot ==========
def plot_bland_altman(npz_path, output_dir, figsize=(8, 6), prefix=''):
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

    save_path = os.path.join(output_dir, f'{prefix}bland_altman.png')
    fig.savefig(save_path)
    plt.close(fig)
    print(f'Saved Bland-Altman plot: {save_path}')


# ========== 3. Age-Group Error Boxplot ==========
def _draw_age_group_boxplot(ax, true, pred, groups, title_suffix=''):
    """Helper: draw a single age-group error boxplot on the given axes."""
    errors_per_group = []
    labels = []
    counts = []

    for g_min, g_max in groups:
        if g_max is None:
            mask = (true >= g_min)
        else:
            mask = (true >= g_min) & (true < g_max)
        if mask.sum() == 0:
            continue
        group_errors = np.abs(pred[mask] - true[mask])
        errors_per_group.append(group_errors)
        label = f'{g_min}-{g_max}' if g_max is not None else f'{g_min}+'
        labels.append(label)
        counts.append(mask.sum())

    if not errors_per_group:
        return

    bp = ax.boxplot(errors_per_group, patch_artist=True, tick_labels=labels,
                    medianprops=dict(color='red', linewidth=1.5),
                    whiskerprops=dict(linewidth=1),
                    boxprops=dict(linewidth=1))

    colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(errors_per_group)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Sample count annotations
    for i, (count, label) in enumerate(zip(counts, labels)):
        ax.text(i + 1, ax.get_ylim()[1] * 0.92, f'n={count}',
                ha='center', va='top', fontsize=9, color='gray')

    # Mean MAE markers
    for i, errs in enumerate(errors_per_group):
        ax.scatter(i + 1, np.mean(errs), marker='D', color='darkred', s=30, zorder=5)

    ax.set_xlabel('Age Group (months)')
    ax.set_ylabel('Absolute Error (months)')
    ax.set_title(f'Prediction Error by Age Group{title_suffix}')
    ax.grid(True, alpha=0.3, axis='y')

    overall_mae = np.mean(np.abs(pred - true))
    ax.axhline(y=overall_mae, color='orange', linestyle='--', linewidth=1,
               label=f'Overall MAE = {overall_mae:.2f}', alpha=0.7)
    ax.legend(loc='upper left')


def plot_age_group_error(npz_path, output_dir, figsize=(9, 6), prefix=''):
    """Generate two age-group error boxplots:
    1. Uniform 3-month intervals: 12-15, 15-18, ..., 27-30 (30+ capped to 30)
    2. Developmental stages: 12-15, 15-18, 18-24, 24-30
    """
    data = load_data(npz_path)
    pred = data['pred_expectations']
    true = data['target_expectations'].copy()

    # Cap 30+ to 30 for grouping
    true_capped = np.clip(true, None, 30)

    # --- Plot 1: uniform 3-month intervals ---
    groups_uniform = [(12, 15), (15, 18), (18, 21), (21, 24), (24, 27), (27, 30)]
    fig1, ax1 = plt.subplots(figsize=figsize)
    _draw_age_group_boxplot(ax1, true_capped, pred, groups_uniform, ' (3-month intervals)')
    save_path1 = os.path.join(output_dir, f'{prefix}age_group_error_uniform.png')
    fig1.savefig(save_path1)
    plt.close(fig1)
    print(f'Saved age-group error plot (uniform): {save_path1}')

    # --- Plot 2: developmental stages ---
    groups_dev = [(12, 15), (15, 18), (18, 24), (24, 30)]
    fig2, ax2 = plt.subplots(figsize=figsize)
    _draw_age_group_boxplot(ax2, true_capped, pred, groups_dev, ' (developmental stages)')
    save_path2 = os.path.join(output_dir, f'{prefix}age_group_error_developmental.png')
    fig2.savefig(save_path2)
    plt.close(fig2)
    print(f'Saved age-group error plot (developmental): {save_path2}')


# ========== 4. LDL Distribution Visualization ==========
def plot_ldl_distribution(npz_path, output_dir, sample_indices=None, figsize=(12, 4)):
    """Visualize predicted vs target label distributions for selected samples.
    
    Args:
        sample_indices: list of sample indices to visualize. Default: auto-select 3 samples
                        (one good, one medium, one poor prediction).
    """
    data = load_data(npz_path)
    
    # Skip if no bin_centers (regression mode, no LDL)
    if 'bin_centers' not in data:
        print('  Skipping LDL distribution plot: no bin_centers (regression mode)')
        return
    
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
    parser.add_argument('--dir', type=str, required=True, help='Directory containing extraction.npz and extraction_retard.npz')
    parser.add_argument('--output_dir', type=str, default='paper_figures/', help='Output directory for figures')
    parser.add_argument('--plots', type=str, nargs='+',
                        default=['scatter', 'scatter_by_case', 'bland_altman', 'age_group', 'ldl', 'umap', 'summary'],
                        help='Which plots to generate')
    parser.add_argument('--samples', type=int, nargs='+', default=None,
                        help='Sample indices for LDL distribution plot')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    npz_path = os.path.join(args.dir, 'extraction.npz')
    npz_retard_path = os.path.join(args.dir, 'extraction_retard.npz')
    
    if not os.path.exists(npz_path):
        print(f"Error: Could not find {npz_path}")
        return
        
    has_retard = os.path.exists(npz_retard_path)

    plot_funcs = {
        'scatter': lambda: plot_scatter(npz_path, args.output_dir, npz_retard_path=npz_retard_path if has_retard else None, agg_by_case=False),
        'scatter_by_case': lambda: plot_scatter(npz_path, args.output_dir, npz_retard_path=npz_retard_path if has_retard else None, agg_by_case=True),
        'bland_altman': lambda: plot_bland_altman(npz_path, args.output_dir),
        'age_group': lambda: plot_age_group_error(npz_path, args.output_dir),
        'ldl': lambda: plot_ldl_distribution(npz_path, args.output_dir, sample_indices=args.samples),
        'umap': lambda: plot_umap(npz_path, args.output_dir),
        'summary': lambda: print_summary(npz_path),
    }

    if has_retard:
        plot_funcs_retard = {
            'bland_altman': lambda: plot_bland_altman(npz_retard_path, args.output_dir, prefix='retard_'),
            'age_group': lambda: plot_age_group_error(npz_retard_path, args.output_dir, prefix='retard_'),
            'summary': lambda: print_summary(npz_retard_path),
        }

    for plot_name in args.plots:
        if plot_name in plot_funcs:
            print(f'\nGenerating: {plot_name}...')
            plot_funcs[plot_name]()
            if has_retard and plot_name in plot_funcs_retard:
                print(f'\nGenerating: retard_{plot_name}...')
                plot_funcs_retard[plot_name]()
        else:
            print(f'Unknown plot type: {plot_name}. Available: {list(plot_funcs.keys())}')

    print(f'\nAll figures saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
