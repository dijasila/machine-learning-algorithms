import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def run_kmeans_clustering(data_path='data/large_dataset.csv', n_clusters=3):
    # Load the dataset
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]  # Use features for clustering, not the target

    # Fit K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Save plot to image folder
    plt.savefig('image/kmeans_clustering_results.png')
    print("K-Means clustering results saved to 'image/kmeans_clustering_results.png'.")

if __name__ == "__main__":
    run_kmeans_clustering()