import numpy as np
from scipy.spatial import cKDTree
import pytest

# Importing the core function
from src.math_utils import generate_curve, l1_loss

def test_generate_curve_output_shape():

    t_array = np.linspace(6, 60, 1500)
    theta_test, M_test, X_test = 30.0, 0.03, 55.0
    
    x, y = generate_curve(t_array, theta_test, M_test, X_test)
    
    assert len(x) == 1500, "X array dimension mismatch"
    assert len(y) == 1500, "Y array dimension mismatch"
    assert isinstance(x, np.ndarray), "Output must be a numpy array for vectorization"

def test_l1_loss_perfect_match():

    t_array = np.linspace(6, 60, 100)
    
    # 1. Generate a "perfect" target curve
    theta_test, M_test, X_test = 30.0, 0.03, 55.0
    x_target, y_target = generate_curve(t_array, theta_test, M_test, X_test)
    
    # 2. Build the spatial KD-Tree from that target
    target_points = np.column_stack((x_target, y_target))
    target_tree = cKDTree(target_points)
    
    # 3. Ask the loss function to evaluate the exact same parameters
    params = [theta_test, M_test, X_test]
    loss = l1_loss(params, target_tree, t_array)
    
    # 4. The L1 spatial distance should be effectively zero
    assert loss < 1e-6, f"Expected perfect match to have 0 error, got {loss}"

def test_l1_loss_spatial_mismatch():

    t_array = np.linspace(6, 60, 100)
    
    # 1. Generate the true target
    x_target, y_target = generate_curve(t_array, 30.0, 0.03, 55.0)
    target_tree = cKDTree(np.column_stack((x_target, y_target)))
    
    # 2. Feed it intentionally bad guesses (e.g., Theta = 10, M = 0)
    bad_params = [10.0, 0.0, 10.0]
    loss = l1_loss(bad_params, target_tree, t_array)
    
    # 3. The error must be significantly higher than zero
    assert loss > 5.0, "Loss function failed to penalize an incorrect spatial mapping"