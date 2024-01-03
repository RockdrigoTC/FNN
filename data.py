import numpy as np
from sklearn.datasets import fetch_openml

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
mnist.target = mnist.target.astype(np.int8)
# MNIST dataset has 70000 samples.

# Split the dataset into training and test sets
print("Splitting the dataset into training and test sets...")
train_size = 60000 # 60000 samples for training and 10000 samples for testing
X_train, X_test, y_train, y_test = mnist.data[:train_size], mnist.data[train_size:], mnist.target[:train_size], mnist.target[train_size:]

# Normalize images to be in the range [0, 1]
print("Normalizing images...")
X_train = X_train / 255.0
X_test = X_test / 255.0

# Encode labels in one-hot format
print("Encoding labels in one-hot format...")
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Save data to numpy files
print("Saving data to numpy files...")
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

# Print dataset information
print("\nDataset information:")
print(f'len(X_train): {len(X_train)} shape: {X_train.shape}')
print(f'len(X_test): {len(X_test)} shape: {X_test.shape}')
print(f'len(y_train): {len(y_train)} shape: {y_train.shape}')
print(f'len(y_test): {len(y_test)} shape: {y_test.shape}')

