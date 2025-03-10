import numpy as np  # Import NumPy for mathematical calculations

class SVM:
    """ Support Vector Machine (SVM) classifier from scratch using Gradient Descent """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate  # Store learning rate
        self.lambda_param = lambda_param  # Store regularization parameter
        self.n_iters = n_iters  # Number of iterations for training
        self.weights = None  # Placeholder for model weights
        self.bias = None  # Placeholder for bias term

    def fit(self, X, y):
        n_samples, n_features = X.shape  # Get number of samples and features
        y = np.where(y <= 0, -1, 1)  # Convert labels from {0,1} to {-1,1}

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)  
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y[idx]))
                    self.bias -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) + self.bias  # Compute decision boundary
        return np.sign(approx)  # Assign class based on sign (+1 or -1)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Generate a simple binary classification dataset
    X, y = make_classification(n_samples=500, n_features=5, n_classes=2, random_state=42)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print the training data
    print("Training Data (Features):\n", X_train[:5])  # Print first 5 rows of features
    print("Training Data (Labels):\n", y_train[:5])  # Print first 5 labels
    
    # Print the test data
    print("Test Data (Features):\n", X_test[:5])  # Print first 5 rows of features
    print("Test Data (Labels):\n", y_test[:5])  # Print first 5 labels

    # Create and train the SVM model
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)

    # Predict on test data
    y_pred = svm.predict(X_test)
    y_pred = np.where(y_pred <= 0, 0, 1)  # Convert labels from {-1,1} to {0,1}

    # Print accuracy of the model
    print("SVM Accuracy:", accuracy_score(y_test, y_pred))
