import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

# Importing our custom math engine
from src.math_utils import l1_loss

def main():
    print("optimising the dataset")
    try:
        df = pd.read_csv('xy_data.csv')

        x_target = df.iloc[:, 0].values
        y_target = df.iloc[:, 1].values
    except FileNotFoundError:
        print("Error: Could not find 'xy_data.csv'.")
        return

    t_target = np.linspace(6, 60, len(df))
    

    bounds = [
        (0.001, 49.999),   # Theta (degrees) bounds
        (-0.049, 0.049),   # M bounds
        (0.001, 99.999)    # X bounds
    ]
    
    print("\nRunning SciPy Global Optimizer (Differential Evolution)")
    print("This will take some time. Let it cook...")
    
    # 3. Run the Optimizer
    result = differential_evolution(
        l1_loss, 
        bounds, 
        args=(t_target, x_target, y_target),
        strategy='best1bin',
        popsize=20,
        tol=1e-6,
        seed=42 # Seed ensures consistent results
    )
    

    best_theta, best_M, best_X = result.x
    

    print("OPTIMIZATION COMPLETE")

    print(f"Final L1 Error: {result.fun:.6f}")
    print(f"Optimal Theta:  {best_theta:.4f} degrees")
    print(f"Optimal M:      {best_M:.4f}")
    print(f"Optimal X:      {best_X:.4f}")

    theta_rad = np.radians(best_theta)
    
    latex_string = (
        f"\\left(t*\\cos({theta_rad:.4f})-e^{{{best_M:.4f}|t|}}\\cdot\\sin(0.3t)\\sin({theta_rad:.4f})+{best_X:.4f}, "
        f"42+t*\\sin({theta_rad:.4f})+e^{{{best_M:.4f}|t|}}\\cdot\\sin(0.3t)\\cos({theta_rad:.4f})\\right)"
    )
    
    print("\nCOPY THIS STRING INTO YOUR README FOR DESMOS:")

    print(latex_string)


if __name__ == "__main__":
    main()