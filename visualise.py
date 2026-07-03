import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Loading xy_data.csv...")
    try:

        df = pd.read_csv('xy_data.csv')
        print(f"Data loaded successfully! Shape: {df.shape}")
        

        print("\nFirst 5 rows of data:")
        print(df.head())

        if 'x' in df.columns.str.lower() and 'y' in df.columns.str.lower():
            x_data = df['x']
            y_data = df['y']
        else:
            x_data = df.iloc[:, 0]
            y_data = df.iloc[:, 1]

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, c='blue', s=15, label='Raw CSV Data', alpha=0.7)
        plt.title('Target Parametric Curve')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        print("\nOpening plot...")
        plt.show()

    except FileNotFoundError:
        print("Error: Could not find 'xy_data.csv'.")

if __name__ == "__main__":
    main()