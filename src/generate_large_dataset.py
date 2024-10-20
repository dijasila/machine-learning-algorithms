import numpy as np
import pandas as pd

def generate_large_dataset(n_samples=100000, n_features=5, output_file='data/large_dataset.csv'):
    # Randomly generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create a linear combination of features as the target with some noise
    coefficients = np.random.rand(n_features)
    y = X.dot(coefficients) + np.random.randn(n_samples) * 0.5  # Adding noise

    # Create a DataFrame and save to CSV
    columns = [f'Feature_{i}' for i in range(1, n_features + 1)]
    data = pd.DataFrame(X, columns=columns)
    data['Target'] = y

    data.to_csv(output_file, index=False)
    print(f"Dataset with {n_samples} samples and {n_features} features created and saved to {output_file}.")

if __name__ == "__main__":
    generate_large_dataset()