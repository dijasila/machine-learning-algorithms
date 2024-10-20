import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def run_pca(data_path='data/large_dataset.csv', n_components=2):
    # Load the dataset
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]  # Exclude label for PCA

    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)

    # Create a DataFrame with principal components
    pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

    # Visualize the first two principal components
    plt.figure(figsize=(10, 6))
    plt.scatter(pc_df['PC1'], pc_df['PC2'], c='blue', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA - First Two Principal Components')

    # Save plot to image folder
    plt.savefig('image/pca_results.png')
    print("PCA results saved to 'image/pca_results.png'.")

if __name__ == "__main__":
    run_pca()