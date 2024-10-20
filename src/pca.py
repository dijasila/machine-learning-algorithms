import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def run_pca(data_path, n_components=2):
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]  # Exclude label for unsupervised PCA

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)

    # Create a DataFrame with principal components
    pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])

    # Plot the first two principal components
    plt.figure(figsize=(8,6))
    plt.scatter(pc_df['PC1'], pc_df['PC2'], c='blue', edgecolor='k', s=40)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA - First Two Principal Components')
    plt.show()

if __name__ == "__main__":
    run_pca('data/large_dataset.csv')