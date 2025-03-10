import numpy as np  # Import NumPy for numerical operations
import random  # Import random module for selecting random features

class DecisionTree:
    """ Manually implemented Decision Tree for both Regression and Classification """

    def __init__(self, max_features=None, min_samples_split=2, max_depth=10, task="regression"):
        """
        Initialize the Decision Tree.

        Parameters:
        - max_features: Number of features to consider at each split (default: sqrt of total features).
        - min_samples_split: Minimum samples required to split a node.
        - max_depth: Maximum depth of the tree.
        - task: "regression" for continuous values, "classification" for categorical labels.
        """
        self.max_features = max_features  # Store max features per split
        self.min_samples_split = min_samples_split  # Minimum number of samples needed for a split
        self.max_depth = max_depth  # Store max depth of the tree
        self.task = task  # Store whether it's regression or classification
        self.tree = None  # Placeholder for the trained tree

    def fit(self, X, y):
        """ Train the decision tree by recursively splitting the dataset """
        self.tree = self._grow_tree(X, y, depth=0)  # Start growing tree from root

    def predict(self, X):
        """ Predict values for each sample in X using the trained tree """
        return np.array([self._traverse_tree(x, self.tree) for x in X])  # Predict for each input sample

    def _grow_tree(self, X, y, depth):
        """ Recursively grow the decision tree """
        n_samples, n_features = X.shape  # Get number of samples and features

        # Stopping conditions for recursion (prevents overfitting)
        if (depth >= self.max_depth or  # Stop if max depth is reached
            n_samples < self.min_samples_split or  # Stop if not enough samples
            len(set(y)) == 1):  # Stop if all labels are the same (pure node)
            return {"value": np.mean(y) if self.task == "regression" else max(set(y), key=list(y).count)}  # Return leaf node value

        # Randomly select a subset of features for better generalization
        feature_indices = random.sample(range(n_features), self.max_features or int(np.sqrt(n_features)))

        # Find the best feature and threshold to split the data
        best_feature, best_threshold = self._find_best_split(X, y, feature_indices)

        # Split dataset based on the chosen feature and threshold
        left_mask = X[:, best_feature] <= best_threshold  # Samples that go to the left child
        right_mask = ~left_mask  # Samples that go to the right child
        left_tree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)  # Grow left subtree
        right_tree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)  # Grow right subtree

        return {"feature": best_feature, "threshold": best_threshold, "left": left_tree, "right": right_tree}  # Return split info

    def _find_best_split(self, X, y, feature_indices):
        """ Find the best feature and threshold to split the dataset """
        best_mse, best_feature, best_threshold = float("inf"), None, None  # Initialize best split values

        for feature in feature_indices:  # Iterate over selected feature subset
            thresholds = np.unique(X[:, feature])  # Get unique values as potential split points
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold  # Identify left group
                right_mask = ~left_mask  # Identify right group

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue  # Skip invalid splits

                mse = self._calculate_mse(y[left_mask], y[right_mask])  # Calculate impurity (MSE)

                if mse < best_mse:  # Update if better split is found
                    best_mse, best_feature, best_threshold = mse, feature, threshold

        return best_feature, best_threshold  # Return best split

    def _calculate_mse(self, y_left, y_right):
        """ Calculate Mean Squared Error for a split """
        def mse(y):
            return np.var(y) * len(y) if len(y) > 0 else 0  # Variance multiplied by sample count
        return mse(y_left) + mse(y_right)  # Sum MSE of both splits

    def _traverse_tree(self, x, node):
        """ Traverse the tree to make predictions """
        if "value" in node:  # If it's a leaf node
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:  # Go left
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])  # Go right


class RandomForest:
    """ Manually implemented Random Forest using multiple decision trees """

    def __init__(self, n_estimators=10, max_features=None, task="regression"):
        """
        Initialize the Random Forest.

        Parameters:
        - n_estimators: Number of trees in the forest.
        - max_features: Number of features per split.
        - task: "regression" or "classification".
        """
        self.n_estimators = n_estimators  # Number of trees
        self.max_features = max_features  # Features per tree
        self.task = task  # Store task type
        self.trees = []  # List of trees

    def fit(self, X, y):
        """ Train multiple decision trees on bootstrap samples """
        self.trees = []  # Reset trees before training
        n_samples = X.shape[0]  # Get number of samples

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)  # Bootstrap Sampling
            X_sample, y_sample = X[indices], y[indices]  # Sample data

            tree = DecisionTree(max_features=self.max_features, task=self.task)  # Create a new tree
            tree.fit(X_sample, y_sample)  # Train tree
            self.trees.append(tree)  # Store trained tree

    def predict(self, X):
        """ Aggregate predictions from all trees """
        predictions = np.array([tree.predict(X) for tree in self.trees])  # Collect predictions from all trees

        if self.task == "regression":
            return np.mean(predictions, axis=0)  # Averaging for regression
        else:
            return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions)  # Majority vote


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, accuracy_score

    # Regression Example
    X_reg, y_reg = make_regression(n_samples=500, n_features=5, noise=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2)

    rf_reg = RandomForest(n_estimators=10, max_features=3, task="regression")
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)

    print("Regression MSE:", mean_squared_error(y_test, y_pred))  # Print Mean Squared Error

    # Classification Example
    X_clf, y_clf = make_classification(n_samples=500, n_features=5, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2)

    rf_clf = RandomForest(n_estimators=10, max_features=3, task="classification")
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)

    print("Classification Accuracy:", accuracy_score(y_test, y_pred))  # Print accuracy score
