import numpy as np  # Import NumPy for numerical computations
import matplotlib.pyplot as plt  # Import Matplotlib for visualization
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Import LDA from scikit-learn

# Define additional data points for Class 1 (red) and Class 2 (blue)
X1 = np.array([
    [3, 1], [2, 5], [1, 4], [4, 7], [5, 3], [2, 2], [6, 1], [3, 2], [4, 4], [5, 5]
])  # Class 1
X2 = np.array([
    [8, 9], [7, 6], [10, 5], [9, 8], [11, 7], [10, 6], [7, 8], [8, 6], [9, 7], [11, 8]
])  # Class 2

# Combine datasets and create labels (0 for Class 1, 1 for Class 2)
X = np.vstack((X1, X2))  
y = np.array([0] * len(X1) + [1] * len(X2))

# Initialize and fit LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

# Project data onto the LDA axis
X_proj = lda.transform(X)

# Scatter plot of LDA projection
plt.scatter(X_proj[:len(X1)], np.zeros(len(X1)), color="red", marker="o", label="Class 1")
plt.scatter(X_proj[len(X1):], np.zeros(len(X2)), color="blue", marker="s", label="Class 2")
plt.axvline(x=0, color="black", linestyle="dashed", label="Decision Boundary")
plt.xlabel("Projected Value")
plt.title("LDA Projection of Data")
plt.legend()
plt.show()

# Predict class for new samples
new_samples = np.array([[5, 5], [7, 7]])  # Added another test sample
predicted_classes = lda.predict(new_samples)

for sample, pred in zip(new_samples, predicted_classes):
    print(f"Predicted class for {sample} is {'Class 2' if pred == 1 else 'Class 1'}")

# Compute class means
mu1, mu2 = np.mean(X1, axis=0), np.mean(X2, axis=0)

# Compute within-class scatter matrix
S1 = np.dot((X1 - mu1).T, (X1 - mu1))  
S2 = np.dot((X2 - mu2).T, (X2 - mu2))  
SW = S1 + S2  

# Compute between-class scatter matrix
SB = np.outer((mu1 - mu2), (mu1 - mu2))

# Compute LDA projection vector
w = np.dot(np.linalg.inv(SW), (mu1 - mu2))  

# Project data onto LDA vector
X1_proj, X2_proj = np.dot(X1, w), np.dot(X2, w)

# Plot original data and LDA direction
plt.scatter(X1[:, 0], X1[:, 1], label="Class 1", color="red", marker="o")
plt.scatter(X2[:, 0], X2[:, 1], label="Class 2", color="blue", marker="s")
plt.plot([0, w[0] * 10], [0, w[1] * 10], color="black", linestyle="dashed", label="LDA Direction")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("LDA Projection Line")
plt.grid()
plt.show()

# Compute decision threshold
z1, z2 = np.dot(mu1, w), np.dot(mu2, w)
z_threshold = (z1 + z2) / 2  

# Scatter plot for 1D projection
plt.scatter(X1_proj, np.zeros_like(X1_proj), color="red", marker="o", label="Class 1")
plt.scatter(X2_proj, np.zeros_like(X2_proj), color="blue", marker="s", label="Class 2")
plt.axvline(x=z_threshold, color="black", linestyle="dashed", label="Decision Boundary")
plt.xlabel("Projected Value")
plt.title("LDA Projection and Decision Boundary")
plt.legend()
plt.show()

# Function to predict class based on LDA projection
def predict_lda(x_new, w, z_threshold):
    z_new = np.dot(x_new, w)  
    return 0 if z_new > z_threshold else 1  

# Example classification with additional new samples
for x_new in new_samples:
    projected_value = np.dot(x_new, w)
    predicted_class = predict_lda(x_new, w, z_threshold)
    print(f"Sample: {x_new}")
    print(f"Projected Value: {projected_value:.3f}")
    print(f"Decision Threshold: {z_threshold:.3f}")
    print(f"Predicted Class: {'Class 2' if predicted_class == 1 else 'Class 1'}\n")
