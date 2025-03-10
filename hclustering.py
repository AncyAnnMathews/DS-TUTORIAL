import numpy as np  # Import NumPy for numerical computations
import matplotlib.pyplot as plt  # Import plotting library
from sklearn.datasets import make_blobs  # Import function to generate sample data

class HierarchicalClustering:
    """ Manually implemented Hierarchical Clustering with Complete Linkage """

    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters  # Store desired number of clusters
        self.clusters = None  # Placeholder for clusters

    def fit(self, X):
        self.clusters = [[i] for i in range(len(X))]  # Create clusters as lists of indices
        distance_matrix = self._compute_distance_matrix(X)  # Compute initial distances

        while len(self.clusters) > self.n_clusters:
            cluster_1, cluster_2 = self._find_closest_clusters(distance_matrix)  # Find closest clusters
            self._merge_clusters(cluster_1, cluster_2, distance_matrix)  # Merge clusters

    def predict(self, X):
        labels = np.zeros(len(X))  # Initialize labels as zeros
        for cluster_id, cluster in enumerate(self.clusters):
            for index in cluster:
                labels[index] = cluster_id  # Assign cluster number to each data point
        return labels

    def _compute_distance_matrix(self, X):
        n_samples = len(X)  # Number of data points
        distance_matrix = np.zeros((n_samples, n_samples))  # Initialize a square matrix
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):  # Compute only upper triangle (symmetric)
                distance = np.linalg.norm(X[i] - X[j])  # Compute Euclidean distance
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix

    def _find_closest_clusters(self, distance_matrix):
        min_distance = float("inf")
        closest_pair = (None, None)

        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                distance = self._complete_linkage_distance(self.clusters[i], self.clusters[j], distance_matrix)
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i, j)

        return closest_pair

    def _complete_linkage_distance(self, cluster1, cluster2, distance_matrix):
        return max(distance_matrix[i, j] for i in cluster1 for j in cluster2)

    def _merge_clusters(self, cluster_1, cluster_2, distance_matrix):
        self.clusters[cluster_1].extend(self.clusters[cluster_2])
        del self.clusters[cluster_2]

        for i in range(len(self.clusters)):
            if i != cluster_1:
                distance_matrix[i, cluster_1] = self._complete_linkage_distance(self.clusters[i], self.clusters[cluster_1], distance_matrix)
                distance_matrix[cluster_1, i] = distance_matrix[i, cluster_1]

        distance_matrix = np.delete(distance_matrix, cluster_2, axis=0)
        distance_matrix = np.delete(distance_matrix, cluster_2, axis=1)


# **Example Usage: Training and Testing the Hierarchical Clustering Model**
if __name__ == "__main__":
    # Generate a synthetic dataset with 3 clusters
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    # Print first 5 rows of data for verification
    print("Sample Data Used:")
    print(X[:5])

    # Create and train the hierarchical clustering model
    hc = HierarchicalClustering(n_clusters=3)
    hc.fit(X)

    # Predict cluster labels
    labels = hc.predict(X)

    # Plot the clustered points
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", edgecolor="k")
    plt.title("Hierarchical Clustering (Complete Linkage)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
