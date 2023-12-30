# FNN(Feedforward Neural Network) implementation
# Author: Rodrigo Torrealba

import numpy as np

# Class definition
class FNN:
    def __init__(self, input_size, hidden_size, output_size, weigth_init="random"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if weigth_init == 'he':
            # Initiliation of weights with He
            scale = np.sqrt(2.0 / input_size)
        elif weigth_init == 'glorot':
            # Initiliation of weights with Glorot
            scale = np.sqrt(2.0 / (input_size + output_size))
        elif weigth_init == 'random':
            scale = 1.0
        else:
            raise ValueError(f"Unknown weigth_init: {weigth_init}")
            
        # Weights and biases initialization
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * scale
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * scale
        self.b2 = np.zeros((1, self.output_size))

    # Activation functions
    def sigmoid(self, x):
        # Sigmoid activation function: 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        # Softmax activation function: e^(x) / sum(e^(x))
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=1, keepdims=True)

    # Forward propagation
    def forward(self, X):
        # Calculate the input to the first layer
        self.z1 = np.dot(X, self.W1) + self.b1
        
        # Apply the sigmoid activation function to the first layer
        self.a1 = self.sigmoid(self.z1)
        
        # Calculate the input to the second layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        # Apply the softmax activation function to the second layer
        self.a2 = self.softmax(self.z2)
        
        # Return the output of the network after forward propagation
        return self.a2


    # Backward propagation
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]

        # Compute the error at the output layer
        self.error = self.a2 - y

        # Compute gradients for the output layer
        self.grad_W2 = np.dot(self.a1.T, self.error) / m
        self.grad_b2 = np.sum(self.error, axis=0) / m

        # Compute the error at the hidden layer
        self.error2 = np.dot(self.error, self.W2.T) * (self.a1 * (1 - self.a1))

        # Compute gradients for the hidden layer
        self.grad_W1 = np.dot(X.T, self.error2) / m
        self.grad_b1 = np.sum(self.error2, axis=0) / m

        # Update weights and biases using gradients and learning rate
        self.W2 -= learning_rate * self.grad_W2
        self.b2 -= learning_rate * self.grad_b2
        self.W1 -= learning_rate * self.grad_W1
        self.b1 -= learning_rate * self.grad_b1

    # Training
    def train(self, X, y, x_test=None, y_test=None, epochs=10, learning_rate=0.01):
        for i in range(epochs):
            print("Epoch #", i)
            y_pred = self.forward(X)
            self.backward(X, y, learning_rate)
            accuracy_train = self.evaluate(X, y)
            loss = self.loss_cross_entropy(y, y_pred)
            if x_test is not None or y_test is not None:
                accuracy_test = self.evaluate(x_test, y_test)
                print(f"Loss: {loss}  Accuracy Train: {accuracy_train} Accuracy Test: {accuracy_test}")
            else:
                print(f"Loss: {loss}  Accuracy Train: {accuracy_train}")

    # Evaluation
    def evaluate(self, X, y):
        # Accuracy calculation: (Number of correct predictions) / (Total number of predictions)
        y_pred = self.forward(X)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return accuracy

    # Prediction
    def predict(self, X):
      return np.round(self.forward(X))

    # Save and load weights and biases
    def saveWeightsBiases(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")
        np.savetxt("b1.txt", self.b1, fmt="%s")
        np.savetxt("b2.txt", self.b2, fmt="%s")

    def loadWeightsBiases(self):
        self.W1 = np.loadtxt("w1.txt", dtype=float)
        self.W2 = np.loadtxt("w2.txt", dtype=float)
        self.b1 = np.loadtxt("b1.txt", dtype=float)
        self.b2 = np.loadtxt("b2.txt", dtype=float)
    
    # Loss function(cross entropy)
    def loss_cross_entropy(self, y, y_pred):
        # Cross entropy loss function: -1/N * sum(y * log(y_pred))
        return -np.mean(y * np.log(y_pred + 1e-10))
    
    # Loss function(mse)
    def loss_mse(self, y, y_pred):
        # Mean squared error loss function formula: 1/N * sum((y - y_pred)^2)
        return np.mean(np.square(y - y_pred))
    
    

