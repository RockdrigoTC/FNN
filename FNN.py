# FNN(Feedforward Neural Network) implementation
# Author: Rodrigo Torrealba

import numpy as np
import pickle

# Layer class definition
class Layer:
    def __init__(self, input_size, output_size, activation_function='sigmoid', weight_init='random'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

        if weight_init == 'he':
            scale = np.sqrt(2.0 / self.input_size)
        elif weight_init == 'glorot':
            scale = np.sqrt(2.0 / (self.input_size + self.output_size))
        elif weight_init == 'random':
            scale = 1.0
        else:
            raise ValueError(f"Unknown weight_init: {weight_init}")

        # Weights and biases initialization
        self.weights = np.random.randn(self.input_size, self.output_size) * scale
        self.biases = np.zeros((1, self.output_size))

    def activate(self, x):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'softmax':
            exps = np.exp(x - np.max(x))
            return exps / np.sum(exps, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_function}")


# Feedforward Neural Network class definition
class FNN:
    def __init__(self, layers=None):
        self.layers = layers
        self.history = {'loss': [], 'accuracyTrain': [], 'accuracyTest': []}

    # Forward propagation
    def forward(self, X):
        activations = X
        for layer in self.layers:
            z = np.dot(activations, layer.weights) + layer.biases
            activations = layer.activate(z)
        return activations


    # Backward propagation
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        activations = X
        layer_inputs = []
        layer_outputs = []

        # Forward pass to store inputs and outputs of each layer
        for layer in self.layers:
            z = np.dot(activations, layer.weights) + layer.biases
            layer_inputs.append(activations)
            activations = layer.activate(z)
            layer_outputs.append(activations)

        # Compute the error at the output layer
        output_error = layer_outputs[-1] - y

        next_layer = None
        # Backward pass to compute gradients for each layer
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            prev_output = layer_inputs[i]
            current_output = layer_outputs[i]

            # Compute gradients for the output layer
            if i == len(self.layers) - 1:
                layer.error = output_error
            else:
                # Compute the error at the hidden layer
                layer.error = np.dot(next_layer.error, next_layer.weights.T) * (current_output * (1 - current_output))

            # Compute gradients for the layer
            layer.grad_weights = np.dot(prev_output.T, layer.error) / m
            layer.grad_biases = np.sum(layer.error, axis=0) / m

            # Update weights and biases using gradients and learning rate
            layer.weights -= learning_rate * layer.grad_weights
            layer.biases -= learning_rate * layer.grad_biases

            # Store the error for the next iteration
            next_layer = layer

    # Training
    def train(self, X=None, y=None, x_test=None, y_test=None, epochs=10, learning_rate=0.01):
        if X is None or y is None:
            raise ValueError("X or y is None")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X and y don't match: {X.shape[0]} != {y.shape[0]}")
        if X.shape[1] != self.layers[0].input_size:
            raise ValueError(f"Input size of the first layer doesn't match: {X.shape[1]} != {self.layers[0].input_size}")
        if y.shape[1] != self.layers[-1].output_size:
            raise ValueError(f"Output size of the last layer doesn't match: {y.shape[1]} != {self.layers[-1].output_size}")
        
        for i in range(epochs):
            print("Epoch #", i)
            y_pred = self.forward(X)
            self.backward(X, y, learning_rate)
            accuracy_train = self.evaluate(X, y)
            loss = self.loss_cross_entropy(y, y_pred)
            if x_test is not None or y_test is not None:
                accuracy_test = self.evaluate(x_test, y_test)
                print(f"Loss: {loss}  Accuracy Train: {accuracy_train} Accuracy Test: {accuracy_test}")
                self.history['accuracyTest'].append(accuracy_test)
            else:
                print(f"Loss: {loss}  Accuracy Train: {accuracy_train}")
            
            self.history['loss'].append(loss)
            self.history['accuracyTrain'].append(accuracy_train)


    # Evaluation
    def evaluate(self, X, y):
        # Accuracy calculation: (Number of correct predictions) / (Total number of predictions)
        y_pred = self.forward(X)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return accuracy

    # Prediction
    def predict(self, X):
      return np.argmax(self.forward(X), axis=1)

    # Loss function(cross entropy)
    def loss_cross_entropy(self, y, y_pred):
        # Cross entropy loss function: -1/N * sum(y * log(y_pred))
        return -np.mean(y * np.log(y_pred + 1e-10))
    
    # Loss function(mse)
    def loss_mse(self, y, y_pred):
        # Mean squared error loss function formula: 1/N * sum((y - y_pred)^2)
        return np.mean(np.square(y - y_pred))
    
    # Save the model to a file
    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    # Load the model from a file
    def load_model(self, filename):
        with open(filename, 'rb') as file:
            loaded_model = pickle.load(file)
        # Copy the loaded model's parameters to the current model
        self.__dict__.update(loaded_model.__dict__)