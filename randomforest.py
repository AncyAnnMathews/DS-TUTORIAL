import numpy as np  # Import NumPy for numerical operations
import random  # Import random module for selecting random features

class DecisionTree:
    """ Manually implemented Decision Tree for both Regression and Classification """

    def __init__(self, max_features=None, min_samples_split=2, max_depth=10, task="regression"):
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.task = task
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        if (depth >= self.max_depth or n_samples < self.min_samples_split or len(set(y)) == 1):
            return {"value": np.mean(y) if self.task == "regression" else max(set(y), key=list(y).count)}

        feature_indices = random.sample(range(n_features), self.max_features or int(np.sqrt(n_features)))
        best_feature, best_threshold = self._find_best_split(X, y, feature_indices)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_tree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return {"feature": best_feature, "threshold": best_threshold, "left": left_tree, "right": right_tree}

    def _find_best_split(self, X, y, feature_indices):
        best_mse, best_feature, best_threshold = float("inf"), None, None

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                mse = self._calculate_mse(y[left_mask], y[right_mask])
                if mse < best_mse:
                    best_mse, best_feature, best_threshold = mse, feature, threshold

        return best_feature, best_threshold

    def _calculate_mse(self, y_left, y_right):
        def mse(y):
            return np.var(y) * len(y) if len(y) > 0 else 0
        return mse(y_left) + mse(y_right)

    def _traverse_tree(self, x, node):
        if "value" in node:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])

class RandomForest:
    """ Manually implemented Random Forest using multiple decision trees """
    def __init__(self, n_estimators=10, max_features=None, task="regression"):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.task = task
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree = DecisionTree(max_features=self.max_features, task=self.task)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        if self.task == "regression":
            return np.mean(predictions, axis=0)
        else:
            return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions)

if __name__ == "__main__":
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, accuracy_score

    # Regression Example
    X_reg, y_reg = make_regression(n_samples=500, n_features=5, noise=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2)
    print("Regression Training Data:")
    print("X_train:", X_train[:5])
    print("y_train:", y_train[:5])
    print("X_test:", X_test[:5])
    print("y_test:", y_test[:5])

    rf_reg = RandomForest(n_estimators=10, max_features=3, task="regression")
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)
    print("Regression MSE:", mean_squared_error(y_test, y_pred))

    # Classification Example
    X_clf, y_clf = make_classification(n_samples=500, n_features=5, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2)
    print("\nClassification Training Data:")
    print("X_train:", X_train[:5])
    print("y_train:", y_train[:5])
    print("X_test:", X_test[:5])
    print("y_test:", y_test[:5])

    rf_clf = RandomForest(n_estimators=10, max_features=3, task="classification")
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    print("Classification Accuracy:", accuracy_score(y_test, y_pred))
