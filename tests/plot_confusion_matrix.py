import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def load_npz(file_path):
    """Load npz file and return the data"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = np.load(file_path)
    return data

def bin_data(data, thresholds):
    """
    Bin data based on thresholds.
    Example thresholds: [12, 18, 24]
    Bins:
    0: thresholds[0] <= x < thresholds[1] (Label: "12-17")
    1: thresholds[1] <= x < thresholds[2] (Label: "18-23")
    ...
    Last bin: thresholds[-1] <= x (Label: "24+")
    """
    binned = np.zeros_like(data, dtype=int) - 1 # Initialize with -1 to catch unbinned
    labels = []
    
    # Sort thresholds just in case
    thresholds = sorted(thresholds)
    
    for i in range(len(thresholds)):
        lower = thresholds[i]
        if i < len(thresholds) - 1:
            upper = thresholds[i+1]
            # Mask for current bin
            mask = (data >= lower) & (data < upper)
            binned[mask] = i
            labels.append(f"{lower}-{upper-1}")
        else:
            # Last bin
            mask = (data >= lower)
            binned[mask] = i
            labels.append(f"{lower}+")
            
    return binned, labels

def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        # Only count if both are within valid bins (>= 0)
        if t >= 0 and p >= 0:
            cm[t, p] += 1
    return cm

def plot_confusion_matrix(cm, labels, title, save_path=None):
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted Label (Bin)')
    plt.ylabel('True Label (Bin)')
    plt.title(title)
    
    if save_path:
        plt.savefig(f"{save_path}_confusion_matrix.png")
        print(f"Plot saved to {save_path}_confusion_matrix.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot Confusion Matrix from npz file with binned labels')
    parser.add_argument('--npz', required=True, help='Path to npz file')
    parser.add_argument('--thresholds', nargs='+', type=int, default=[12, 18, 24], 
                        help='List of thresholds for binning (e.g. 12 18 24)')
    parser.add_argument('--save_plot', help='Path to save the plot')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.npz}...")
    data = load_npz(args.npz)
    
    # Get data
    # Assuming raw class_label is 0-based and needs +12 to match the 12-36 range description
    # similar to plot_ldl_labels.py
    if 'class_label' in data:
        raw_true = data['class_label'] + 12
        print("Class Labels (adjusted):", raw_true)
    else:
        print("Error: 'class_label' not found in npz file.")
        return

    if 'pred_expectations' in data:
        raw_pred = data['pred_expectations']
        print("Predicted Expectations:", raw_pred)
    else:
        print("Error: 'pred_expectations' not found in npz file.")
        return
    
    # Round predictions as requested
    raw_pred_rounded = np.round(raw_pred)
    
    print(f"Using thresholds: {args.thresholds}")
    
    # Bin data
    binned_true, bin_labels = bin_data(raw_true, args.thresholds)
    binned_pred, _ = bin_data(raw_pred_rounded, args.thresholds)
    
    # Check for unbinned data (values smaller than the first threshold)
    if np.any(binned_true == -1):
        print(f"Warning: {np.sum(binned_true == -1)} true labels are smaller than the minimum threshold {min(args.thresholds)} and will be ignored.")
    if np.any(binned_pred == -1):
        print(f"Warning: {np.sum(binned_pred == -1)} predicted labels are smaller than the minimum threshold {min(args.thresholds)} and will be ignored.")

    # Compute CM
    cm = compute_confusion_matrix(binned_true, binned_pred, len(bin_labels))
    
    # Plot
    plot_confusion_matrix(cm, bin_labels, f'Confusion Matrix\nThresholds: {args.thresholds}', args.save_plot)

if __name__ == '__main__':
    main()