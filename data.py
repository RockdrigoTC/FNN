import numpy as np
from sklearn.datasets import fetch_openml

dataset_name = 'mnist_784'
# Options:
#   - mnist_784
#   - fashion-mnist
#   - diabetes
#   - iris
#   - wine

"""
You can also use other datasets. But probably you will need to do some preprocessing.
"""

# Load MNIST fashion dataset
print(f"Loading {dataset_name} dataset...")
X, y = fetch_openml(dataset_name, version=1, return_X_y=True, parser='auto', as_frame=False)

# Dataset information
print("\nDataset information:")
print(f'Number of samples: {X.shape[0]}')
print(f'Number of features: {X.shape[1]}')
print(f'Number of classes: {len(np.unique(y))}')
classes = np.unique(y, return_counts=True)
for i in range(len(classes[0])):
    print(f'   Class {classes[0][i]}: {classes[1][i]} samples')

# Convert string labels to numerical values
if isinstance(y[0], str):
    print("\nConverting string labels to numerical values...")
    label_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
    y = np.array([label_mapping[label] for label in y])
    print(f'Mapping: {label_mapping}\n')

# Shuffle data
print("Shuffling data...")
shuffle_idx = np.arange(X.shape[0])
np.random.shuffle(shuffle_idx)
X = X[shuffle_idx]
y = y[shuffle_idx]

# 90% train, 10% test
train_size = int(len(X) * 0.9)

# Split data into train and test sets
print("Splitting data into train and test sets...")
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normalize images to be in the range [0, 1]
print("Normalizing images...")
X_train = X_train / 255.0
X_test = X_test / 255.0

# Encode labels in one-hot format
print("Encoding labels in one-hot format...")
num_classes = len(np.unique(y_train))
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Save data to numpy files
print("Saving data to numpy files...")
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

# Data information
print("\nData information:")
print(f'X_train -> shape: {X_train.shape}')
print(f'X_test  -> shape: {X_test.shape}')
print(f'y_train -> shape: {y_train.shape}')
print(f'y_test  -> shape: {y_test.shape}')