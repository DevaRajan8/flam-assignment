import numpy as np

def generate_curve(t, theta_deg, M, X):

    theta_rad = np.radians(theta_deg)

    exp_term = np.exp(M * np.abs(t))
    
    # The core equations from the PDF
    x = t * np.cos(theta_rad) - exp_term * np.sin(0.3 * t) * np.sin(theta_rad) + X
    y = 42 + t * np.sin(theta_rad) + exp_term * np.sin(0.3 * t) * np.cos(theta_rad)
    
    return x, y

def l1_loss(params, t_target, x_target, y_target):

    theta_deg, M, X = params
    x_pred, y_pred = generate_curve(t_target, theta_deg, M, X)

    error_x = np.mean(np.abs(x_pred - x_target))
    error_y = np.mean(np.abs(y_pred - y_target))

    return error_x + error_y