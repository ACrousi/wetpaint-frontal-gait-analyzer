import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import os

def load_npz(file_path):
    """Load npz file and return the data"""
    data = np.load(file_path)
    return data

def calculate_std(pred_distribution, bin_centers, pred_age):
    """
    Calculate prediction uncertainty (Standard Deviation).
    
    Args:
        pred_distribution (np.array): Softmax output (N, ClassNum)
        bin_centers (np.array): Age values for each class (ClassNum,)
        pred_age (np.array): Expected age predictions (N,)
        
    Returns:
        np.array: Standard deviation for each prediction (N,)
    """
    # Ensure shapes align for broadcasting
    # pred_distribution: (N, C)
    # bin_centers: (C,) -> (1, C)
    # pred_age: (N,) -> (N, 1)
    
    if len(bin_centers.shape) == 1:
        bin_centers = bin_centers.reshape(1, -1)
    
    if len(pred_age.shape) == 1:
        pred_age = pred_age.reshape(-1, 1)
        
    # Calculate variance: sum(p * (x - mu)^2)
    # (N, C) * ( (1, C) - (N, 1) )^2
    variance = np.sum(pred_distribution * (bin_centers - pred_age)**2, axis=1)
    std_dev = np.sqrt(variance)
    
    return std_dev

def get_case_id(file_path):
    """Extract CASEID from file path (first two parts before '_')"""
    filename = file_path.split('\\')[-1].split('.')[0]
    parts = filename.split('_')
    return parts[0] + '_' + parts[1]

def get_case_id_from_name(name):
    """Extract CASEID from name (first two parts before '_')"""
    # Handle bytes if necessary
    if isinstance(name, bytes):
        name = name.decode('utf-8')
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0] + '_' + parts[1]
    return name

def group_data(names, labels, preds):
    """Group data by CASEID and calculate average predictions"""
    groups = defaultdict(list)
    for name, label, pred in zip(names, labels, preds):
        case_id = get_case_id_from_name(name)
        groups[case_id].append((label, pred))
    
    grouped_labels = []
    grouped_preds = []
    
    for case_id, values in groups.items():
        # values is list of (label, pred)
        vals = np.array(values)
        # Average labels and preds
        avg_label = np.mean(vals[:, 0])
        avg_pred = np.mean(vals[:, 1])
        
        grouped_labels.append(avg_label)
        grouped_preds.append(avg_pred)
        
    return np.array(grouped_labels), np.array(grouped_preds)


def plot_boxplot(data_list, labels, title, save_path=None):
    """Plot box plot for given data"""
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_list, labels=labels)
    plt.title(title)
    plt.ylabel('Value')
    plt.grid(True, axis='y')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_scatter(class_labels1, pred_expectations1, class_labels2, pred_expectations2, title, save_path=None):
    """Plot scatter plot with class_label as x and pred_expectations as y for both datasets"""
    plt.figure(figsize=(10, 6))
    plt.scatter(class_labels1, pred_expectations1, alpha=0.6, color='blue', label='NPZ1')
    plt.scatter(class_labels2, pred_expectations2, alpha=0.6, color='red', label='NPZ2')

    # Fit and plot regression line for NPZ1
    if len(class_labels1) > 1:
        coeff1 = np.polyfit(class_labels1, pred_expectations1, 1)
        poly1 = np.poly1d(coeff1)
        x_fit1 = np.linspace(class_labels1.min(), class_labels1.max(), 100)
        plt.plot(x_fit1, poly1(x_fit1), color='blue', linestyle='--', linewidth=2, label='NPZ1 Regression')

    # Fit and plot regression line for NPZ2
    if len(class_labels2) > 1:
        coeff2 = np.polyfit(class_labels2, pred_expectations2, 1)
        poly2 = np.poly1d(coeff2)
        x_fit2 = np.linspace(class_labels2.min(), class_labels2.max(), 100)
        plt.plot(x_fit2, poly2(x_fit2), color='red', linestyle='--', linewidth=2, label='NPZ2 Regression')

    plt.xlabel('Class Label')
    plt.ylabel('Predicted Expectations')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_std_scatter(true_age1, std_dev1, true_age2, std_dev2, title, save_path=None):
    """Plot scatter plot of Prediction Std Dev vs True Age"""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(true_age1, std_dev1, alpha=0.6, color='blue', label='NPZ1 (Normal)')
    plt.scatter(true_age2, std_dev2, alpha=0.6, color='red', label='NPZ2 (Retarded)')
    
    plt.xlabel('True Age (Months)')
    plt.ylabel('Prediction Std Dev (Months)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_pairwise_consistency(true_labels, pred_labels, title, save_path=None, n_bins=10):
    """
    Plot Pairwise Consistency: X-axis = |GT difference|, Y-axis = Pairwise correctness.
    
    For each pair (i, j), pairwise correctness is 1 if:
      - sign(pred_i - pred_j) == sign(gt_i - gt_j) (correct ordering)
    Otherwise 0.
    
    We bin pairs by |gt_i - gt_j| and compute the average correctness per bin.
    
    Args:
        true_labels (np.array): Ground truth labels (N,)
        pred_labels (np.array): Predicted labels/expectations (N,)
        title (str): Plot title
        save_path (str): Path to save the plot
        n_bins (int): Number of bins for GT difference
    """
    n = len(true_labels)
    
    # Calculate all pairwise differences and correctness
    gt_diffs = []
    correctness = []
    
    for i in range(n):
        for j in range(i + 1, n):
            gt_diff = true_labels[i] - true_labels[j]
            pred_diff = pred_labels[i] - pred_labels[j]
            
            abs_gt_diff = np.abs(gt_diff)
            
            # Skip pairs with zero GT difference (tied pairs)
            if abs_gt_diff < 1e-6:
                continue
            
            # Correctness: 1 if prediction preserves the ordering, 0 otherwise
            is_correct = 1 if (gt_diff * pred_diff > 0) else 0
            
            gt_diffs.append(abs_gt_diff)
            correctness.append(is_correct)
    
    gt_diffs = np.array(gt_diffs)
    correctness = np.array(correctness)
    
    if len(gt_diffs) == 0:
        print("No valid pairs found for pairwise consistency plot.")
        return
    
    # Create bins based on GT difference
    bin_edges = np.linspace(gt_diffs.min(), gt_diffs.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate average correctness per bin
    bin_correctness = []
    bin_counts = []
    bin_stds = []
    
    for i in range(n_bins):
        mask = (gt_diffs >= bin_edges[i]) & (gt_diffs < bin_edges[i + 1])
        if i == n_bins - 1:  # Include the last edge
            mask = (gt_diffs >= bin_edges[i]) & (gt_diffs <= bin_edges[i + 1])
        
        if np.sum(mask) > 0:
            bin_correctness.append(np.mean(correctness[mask]))
            bin_counts.append(np.sum(mask))
            bin_stds.append(np.std(correctness[mask]))
        else:
            bin_correctness.append(np.nan)
            bin_counts.append(0)
            bin_stds.append(0)
    
    bin_correctness = np.array(bin_correctness)
    bin_counts = np.array(bin_counts)
    bin_stds = np.array(bin_stds)
    
    # Calculate standard error for error bars
    bin_se = bin_stds / np.sqrt(np.maximum(bin_counts, 1))
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Primary axis: Pairwise correctness
    valid_mask = ~np.isnan(bin_correctness)
    ax1.errorbar(bin_centers[valid_mask], bin_correctness[valid_mask], 
                 yerr=bin_se[valid_mask], fmt='o-', color='blue', 
                 linewidth=2, markersize=8, capsize=4, label='Pairwise Correctness')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Baseline (0.5)')
    
    ax1.set_xlabel('|GT Difference| (Months)', fontsize=12)
    ax1.set_ylabel('Pairwise Correctness', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis: Number of pairs per bin
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, bin_counts, width=(bin_edges[1] - bin_edges[0]) * 0.8, 
            alpha=0.3, color='orange', label='Pair Count')
    ax2.set_ylabel('Number of Pairs', fontsize=12, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    
    # Print summary statistics
    overall_correctness = np.mean(correctness)
    print(f"\n=== Pairwise Consistency Summary ===")
    print(f"Total valid pairs: {len(correctness)}")
    print(f"Overall pairwise correctness: {overall_correctness:.4f}")
    print(f"(Random baseline: 0.5)")
    
    return overall_correctness

def main():
    parser = argparse.ArgumentParser(description='Plot LDL labels from npz files')
    parser.add_argument('--npz1', required=True, help='Path to first npz file')
    parser.add_argument('--npz2', required=True, help='Path to second npz file')
    parser.add_argument('--save_plot', help='Path to save the plot')

    args = parser.parse_args()

    # Load data
    data1 = load_npz(args.npz1)
    data2 = load_npz(args.npz2)

    # Get predicted expectations
    pred_expectations1 = data1['pred_expectations']
    print("Predicted Expectations from first npz:", pred_expectations1)
    print("Std of Predicted Expectations from first npz:", np.std(pred_expectations1))
    pred_expectations2 = data2['pred_expectations']
    print("Predicted Expectations from second npz:", pred_expectations2)
    print("Std of Predicted Expectations from second npz:", np.std(pred_expectations2))

    # Get class labels using original target_expectations (not bin_centers)
    class_labels1 = data1['target_expectations']
    print("Class Labels from first npz (using target_expectations):", class_labels1)
    print("Std of Class Labels from first npz:", np.std(class_labels1))
    class_labels2 = data2['target_expectations']
    print("Class Labels from second npz (using target_expectations):", class_labels2)
    print("Std of Class Labels from second npz:", np.std(class_labels2))

    # Plot original datasets on the same plot
    plot_scatter(class_labels1, pred_expectations1, class_labels2, pred_expectations2, f'Combined: Noraml vs Retard', args.save_plot)

    # Group data by CASEID and plot
    names1 = data1['name']
    names2 = data2['name']
    
    print("Grouping data by CASEID...")
    g_labels1, g_preds1 = group_data(names1, class_labels1, pred_expectations1)
    g_labels2, g_preds2 = group_data(names2, class_labels2, pred_expectations2)

    print("Group label std:", np.std(g_labels1))
    print("Group predicted std:", np.std(g_preds1))
    print("Group label std:", np.std(g_labels2))
    print("Group predicted std:", np.std(g_preds2))
    
    save_path_grouped = None
    if args.save_plot:
        base, ext = os.path.splitext(args.save_plot)
        save_path_grouped = f"{base}_grouped{ext}"
    
    plot_scatter(g_labels1, g_preds1, g_labels2, g_preds2, f'Combined Grouped:  Noraml vs Retard', save_path_grouped)

    # Plot box plots for data1
    save_path_box1 = None
    if args.save_plot:
        base, ext = os.path.splitext(args.save_plot)
        save_path_box1 = f"{base}_boxplot_data1{ext}"
    
    plot_boxplot([class_labels1, pred_expectations1],
                 ['Class Label', 'Predicted Expectations'],
                 f'Box Plot Data 1: {args.npz1}',
                 save_path_box1)

    # Plot box plots for data2
    save_path_box2 = None
    if args.save_plot:
        base, ext = os.path.splitext(args.save_plot)
        save_path_box2 = f"{base}_boxplot_data2{ext}"

    plot_boxplot([class_labels2, pred_expectations2],
                 ['Class Label', 'Predicted Expectations'],
                 f'Box Plot Data 2: {args.npz2}',
                 save_path_box2)

    # ---------------------------------------------------------
    # New Plot: Prediction Std/Entropy vs True Age
    # ---------------------------------------------------------
    print("\nCalculating and Plotting Prediction Uncertainty...")
    
    # helper to get std
    def get_uncertainty_data(data, pred_expectations, class_labels_adj):
        # We need bin_centers and out (softmax)
        # If 'out' is available in npz.
        if 'out' not in data or 'bin_centers' not in data:
            print("Warning: 'out' or 'bin_centers' not found in npz. Cannot calculate Std.")
            return None, None
        
        out_dist = data['out']
        bin_centers = data['bin_centers']
        
        # Calculate std
        std_vals = calculate_std(out_dist, bin_centers, pred_expectations)
        
        # Determine True Age
        # User defined Class Label + 12 as X in original code.
        # But for 'True Age', target_expectations is more accurate if available.
        # We'll default to target_expectations if available, else class_labels_adj.
        if 'target_expectations' in data:
            true_age = data['target_expectations']
        else:
            true_age = class_labels_adj # Fallback
            
        return true_age, std_vals

    true_age1, std1 = get_uncertainty_data(data1, pred_expectations1, class_labels1)
    true_age2, std2 = get_uncertainty_data(data2, pred_expectations2, class_labels2)
    
    if std1 is not None and std2 is not None:
        save_path_std = None
        if args.save_plot:
            base, ext = os.path.splitext(args.save_plot)
            save_path_std = f"{base}_uncertainty{ext}"
            
        plot_std_scatter(true_age1, std1, true_age2, std2,
                         'Prediction Uncertainty vs True Age',
                         save_path_std)

    # ---------------------------------------------------------
    # New Plot: Pairwise Consistency Plot
    # ---------------------------------------------------------
    print("\nPlotting Pairwise Consistency...")
    
    # Plot for NPZ1
    save_path_pairwise1 = None
    if args.save_plot:
        base, ext = os.path.splitext(args.save_plot)
        save_path_pairwise1 = f"{base}_pairwise_npz1{ext}"
    
    plot_pairwise_consistency(class_labels1, pred_expectations1,
                              f'Pairwise Consistency: NPZ1 (Normal)',
                              save_path_pairwise1)
    
    # Plot for NPZ2
    save_path_pairwise2 = None
    if args.save_plot:
        base, ext = os.path.splitext(args.save_plot)
        save_path_pairwise2 = f"{base}_pairwise_npz2{ext}"
    
    plot_pairwise_consistency(class_labels2, pred_expectations2,
                              f'Pairwise Consistency: NPZ2 (Retarded)',
                              save_path_pairwise2)
    
    # Combined plot (all samples)
    combined_labels = np.concatenate([class_labels1, class_labels2])
    combined_preds = np.concatenate([pred_expectations1, pred_expectations2])
    
    save_path_pairwise_combined = None
    if args.save_plot:
        base, ext = os.path.splitext(args.save_plot)
        save_path_pairwise_combined = f"{base}_pairwise_combined{ext}"
    
    plot_pairwise_consistency(combined_labels, combined_preds,
                              'Pairwise Consistency: Combined (All Samples)',
                              save_path_pairwise_combined)

if __name__ == '__main__':
    main()