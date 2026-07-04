import pandas as pd
import numpy as np
# Differential Evolution is a global optimizer, preventing us from getting stuck in sinusoidal local minima.
from scipy.optimize import differential_evolution
# KDTree allows us to calculate spatial distance regardless of how the target CSV is shuffled.
from scipy.spatial import cKDTree

# Importing our custom mathematical loss function and generator.
from math_utils import l1_loss

def main():
    print("optimising the dataset")
    try:
        # Loading the raw CSV file provided by the assignment.
        df = pd.read_csv('xy_data.csv')
        
        # Extracting the target X values from the first column.
        x_target = df.iloc[:, 0].values
        
        # Extracting the target Y values from the second column.
        y_target = df.iloc[:, 1].values
        
    except FileNotFoundError:
        # Fail-safe in case the script is run from the wrong directory.
        print("Error: Could not find 'xy_data.csv'.")
        return

    # Generating the independent variable 't' as 1,500 evenly spaced steps from 6 to 60.
    t_target = np.linspace(6, 60, len(df))
    
    # Stacking the raw, shuffled target data into physical [x, y] coordinate pairs.
    target_points = np.column_stack((x_target, y_target))
    
    # Building the spatial tree structure in memory so the optimizer can query it rapidly.
    target_tree = cKDTree(target_points)

    # Defining the strict search space boundaries provided in the assignment documentation.
    bounds = [
        (0.001, 49.999),   # Theta (degrees) bounds
        (-0.049, 0.049),   # M bounds
        (0.001, 99.999)    # X bounds
    ]
    
    print("\nRunning SciPy Global Optimizer (Differential Evolution)")
    print("This will take some time. Let it cook...")
    
    # Executing the differential evolution solver.
    result = differential_evolution(
        l1_loss, # The function we are trying to minimize.
        bounds, # The search boundaries for our three variables.
        args=(target_tree, t_target), # Passing our spatial tree and time array to the loss function.
        strategy='best1bin', # The standard mutation strategy for differential evolution.
        popsize=20, # The number of candidate solutions generated per generation.
        tol=1e-6, # The convergence tolerance before the optimizer stops.
        seed=42 # Locking the random seed ensures the results are completely reproducible.
    )
    
    # Unpacking the final, optimal parameters found by the algorithm.
    best_theta, best_M, best_X = result.x
    
    print("OPTIMIZATION COMPLETE")
    
    # Printing the final error to verify the curve mapped correctly (should be near 0.026).
    print(f"Final L1 Error: {result.fun:.6f}")
    
    # Displaying the optimal numbers to be documented in the README.
    print(f"Optimal Theta:  {best_theta:.4f} degrees")
    print(f"Optimal M:      {best_M:.4f}")
    print(f"Optimal X:      {best_X:.4f}")

    # Converting the final optimal theta to radians for the final LaTeX string generation.
    theta_rad = np.radians(best_theta)
    
    # Dynamically injecting the optimal variables into a plain-text string formatted perfectly for Desmos.
    latex_string = (
        f"(t * \\cos({theta_rad:.4f}) - e^({best_M:.4f} * |t|) * \\sin(0.3 * t) * \\sin({theta_rad:.4f}) + {best_X:.4f}, "
        f"42 + t * \\sin({theta_rad:.4f}) + e^({best_M:.4f} * |t|) * \\sin(0.3 * t) * \\cos({theta_rad:.4f}))"
    )
    
    # Outputting the final payload.
    print(latex_string)


if __name__ == "__main__":
    main()