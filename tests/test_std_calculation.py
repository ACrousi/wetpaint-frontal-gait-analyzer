import numpy as np
import sys
import os

# Add current directory to path so we can import the module
sys.path.append(os.getcwd())

from plot_ldl_labels import calculate_std

def test_std_calculation():
    print("Testing calculate_std function...")
    
    # Case 1: Perfect prediction (deterministic)
    # Prob: [0, 1, 0], Age: [10, 20, 30], PredAge: 20
    probs = np.array([[0, 1, 0]])
    bins = np.array([10, 20, 30])
    pred_age = np.array([20])
    
    std = calculate_std(probs, bins, pred_age)
    print(f"Case 1 (Deterministic): Expected 0.0, Got {std[0]}")
    assert np.isclose(std[0], 0.0), f"Case 1 failed: {std[0]}"
    
    # Case 2: Uniform distribution
    # Prob: [0.33, 0.33, 0.33], Age: [19, 20, 21], PredAge: 20
    # Var = (-1)^2 * 1/3 + (0)^2 * 1/3 + (1)^2 * 1/3 = 2/3 = 0.666
    # Std = sqrt(0.666) ~= 0.816
    probs = np.array([[1/3, 1/3, 1/3]])
    bins = np.array([19, 20, 21])
    pred_age = np.array([20])
    
    std = calculate_std(probs, bins, pred_age)
    print(f"Case 2 (Uniform): Expected ~0.816, Got {std[0]}")
    assert np.isclose(std[0], np.sqrt(2/3)), f"Case 2 failed: {std[0]}"
    
    # Case 3: Two peaks
    # Prob: [0.5, 0, 0.5], Age: [10, 20, 30], PredAge: 20
    # Var = (10-20)^2 * 0.5 + (30-20)^2 * 0.5 = 100*0.5 + 100*0.5 = 100
    # Std = 10
    probs = np.array([[0.5, 0, 0.5]])
    bins = np.array([10, 20, 30])
    pred_age = np.array([20])
    
    std = calculate_std(probs, bins, pred_age)
    print(f"Case 3 (Two Peaks): Expected 10.0, Got {std[0]}")
    assert np.isclose(std[0], 10.0), f"Case 3 failed: {std[0]}"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_std_calculation()
