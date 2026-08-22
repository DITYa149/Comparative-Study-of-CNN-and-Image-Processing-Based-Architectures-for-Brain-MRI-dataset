import numpy as np

# ----------- LOAD DATA -----------

data = np.loadtxt("features.csv", delimiter=",", skiprows=1)

X = data[:, :-1]
y = data[:, -1]

# Store normalization parameters for later use
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

# Normalize features (important!)
X = (X - X_mean) / X_std

# Add bias term
bias = np.ones((X.shape[0], 1))
X = np.hstack((bias, X))

# Initialize weights
np.random.seed(42)
weights = np.random.randn(X.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow

# Training parameters
learning_rate = 0.1
epochs = 1000

# ----------- TRAINING LOOP -----------

for epoch in range(epochs):

    z = np.dot(X, weights)
    predictions = sigmoid(z)

    error = y - predictions
    gradient = np.dot(X.T, error) / len(y)

    weights += learning_rate * gradient

    if epoch % 100 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nTraining Complete!")

# ----------- TEST ACCURACY -----------

predictions = sigmoid(np.dot(X, weights))
predicted_classes = (predictions >= 0.5).astype(int)

accuracy = np.mean(predicted_classes == y) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Save weights AND normalization parameters
np.save("trained_weights.npy", weights)
np.save("feature_mean.npy", X_mean)
np.save("feature_std.npy", X_std)

print("Model weights and normalization parameters saved.")