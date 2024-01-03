import numpy as np
import matplotlib.pyplot as plt
from FNN import Layer, FNN

# Load data from numpy files
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Data information
input = X_train.shape[1]     # Size of input layer
output = y_train.shape[1]    # Size of output layer

# Hyperparameters
hidden_size = 100            # Size of hidden layer
epoch = 10                   # Number of epochs
batch_size = 10000           # Batch size
learning_rate = 0.1          # Learning rate

# Crear capas
layer_1 = Layer(input_size=input, output_size=hidden_size, activation='tanh', weight_init='he')
layer_2 = Layer(input_size=hidden_size, output_size=output, activation='softmax', weight_init='he')

"""
Try different hyperparameters, architectures, activation functions, weight initialization methods, etc. See how they affect the model performance.
Examples:
    - Change the number of hidden layers and their sizes: hidden_size = 50, hidden_size = 150, hidden_size = 200, etc.
    - Change the activation functions: activation='sigmoid', activation='tanh', activation='relu', activation='softmax', etc.
    - Change the weight initialization methods: weight_init='random', weight_init='he', weight_init='glorot', etc.
    - Change the learning rate: learning_rate = 0.1, learning_rate = 0.5, learning_rate = 0.01, etc.
    - Change the number of epochs: epoch = 10, epoch = 100, epoch = 200, etc.
    - Change the batch size: batch_size = 32, batch_size = 64, batch_size = 1000. batch_size = 10000, etc.
"""

# Create and train the model
model = FNN(layers=[layer_1, layer_2])
model.train(X=X_train, y=y_train, x_test=X_test, y_test=y_test , epochs=epoch, learning_rate=learning_rate, batch_size=batch_size)

# Save the model
model.save_model("model.pkl")

# Load the model
model.load_model("model.pkl")

# Single prediction
predict = model.predict(X_test[0]).astype(int)
real_class = np.argmax(y_test[0]).astype(int)
print(f"Predicted class: {predict}")
print(f"Real class: {real_class}")
print("-----------")

# Multiple predictions
sample_indices = np.random.choice(len(X_test), size=10, replace=False)
for i in sample_indices:
    predict = model.predict(X_test[i]).astype(int)
    real_class = np.argmax(y_test[i]).astype(int)
    print(f"Predicted class: {predict}")
    print(f"Real class: {real_class}")
    print("-----------")


# Plot loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(model.history['loss'])
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(122)
plt.plot(model.history['accuracyTrain'], label='train')
plt.plot(model.history['accuracyTest'], label='test')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

