import numpy as np
# Importing cKDTree to map our data as a physical 2D point cloud, bypassing row order entirely.

def generate_curve(t, theta_deg, M, X):
    """Generates the curve and returns an array of (x, y) coordinates."""
    
    # The assignment provides theta in degrees, but numpy requires radians for trigonometry.
    theta_rad = np.radians(theta_deg)
    
    # Pre-computing the exponential growth term to optimize execution speed.
    exp_term = np.exp(M * np.abs(t))
    
    # Constructing the X-coordinate using the core parametric equation from the PDF.
    x = t * np.cos(theta_rad) - exp_term * np.sin(0.3 * t) * np.sin(theta_rad) + X
    
    # Constructing the Y-coordinate using the core parametric equation from the PDF.
    y = 42 + t * np.sin(theta_rad) + exp_term * np.sin(0.3 * t) * np.cos(theta_rad)
    
    # Stacking the separate X and Y arrays into a single array of [x, y] coordinate pairs.
    return x, y

def l1_loss(params, target_tree, t_array):
    """Calculates L1 distance using a KD-Tree to ignore data shuffling."""
    
    # Unpacking the current guess from the optimizer.
    theta_deg, M, X = params
    
    # Generating our predicted curve using the current variable guesses.
    x_pred, y_pred = generate_curve(t_array, theta_deg, M, X)
    
    # Formatting our predictions into a 2D coordinate matrix: [[x1, y1], [x2, y2], ...]
    pred_points = np.column_stack((x_pred, y_pred))
    
    # The KD-Tree searches spatial geometry to find the true closest target point for every predicted point.
    # p=1 forces the tree to calculate the Manhattan (L1) distance, matching the assignment's criteria.
    distances, _ = target_tree.query(pred_points, p=1)
    
    # Returning the average spatial error across all 1,500 points. The optimizer tries to drive this to zero.
    return np.mean(distances)