import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def run_kmeans_clustering(data_path, n_clusters=3):
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]  # No labels for clustering

    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_predict(X)

    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
    plt.title('K-Means Clustering')
    plt.show()

if __name__ == "__main__":
    run_kmeans_clustering('data/large_dataset.csv')