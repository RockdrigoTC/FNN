import numpy as np
import pickle

# Layer class definition
class Layer:
    def __init__(self, input_size=None, output_size=None, activation='sigmoid', weight_init='random'):
        """
        Layer class constructor
        Args:
            input_size (int)             : Size of the input layer
            output_size (int)            : Size of the output layer
            activation_function (string) : Activation function
            weight_init (string)         : Weight initialization method
        """

        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation

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
        """
        Activation function
        Args:
            x (ndarray)  : Input data

        Returns:
            ndarray      : Output data
        """
        if self.activation_function == 'sigmoid':
            # Sigmoid function: 1 / (1 + e^(-x))
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'softmax':
            # Softmax function: e^(x - max(x)) / sum(e^(x - max(x)))
            exps = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)
        elif self.activation_function == 'relu':
            # ReLU function: max(0, x)
            return np.maximum(0, x)
        elif self.activation_function == 'tanh':
            # Tanh function: (e^x - e^(-x)) / (e^x + e^(-x))
            return np.tanh(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_function}")


# Feedforward Neural Network class definition
class FNN:
    def __init__(self, layers=None):
        """
        FNN class constructor
        Args:
            layers (list) : List of Layer objects
        """
        self.layers = layers
        self.history = {'loss': [], 'accuracyTrain': [], 'accuracyTest': []}

    def forward(self, X):
        """
        Forward propagation
        Args:
            X (ndarray(m, n))      : Input data

        Returns:
            activations (ndarray)  : Output data
        """

        activations = X
        for layer in self.layers:
            z = np.dot(activations, layer.weights) + layer.biases
            activations = layer.activate(z)
        return activations

    def backward(self, X, y, learning_rate=0.01):
        """
        Backpropagation
        Args:
            X (ndarray(m, n))      : Input data
            y (ndarray(m, ))       : Target data
            learning_rate (float)  : Learning rate
        """

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

    def train(self, X=None, y=None, x_test=None, y_test=None, epochs=10, learning_rate=0.01, batch_size=None):
        """
        Training
        Args:
            X (ndarray(m, n))      : Input data
            y (ndarray(m, ))       : Target data
            x_test (ndarray(m, n)) : Input data for testing (Optional)
            y_test (ndarray(m, ))  : Target data for testing (Optional)
            epochs (int)           : Number of epochs (Number of times the entire dataset is passed forward and backward through the neural network)
            learning_rate (float)  : Learning rate (alpha)
            batch_size (int)       : Batch size (Number of samples per gradient update)
        """

        if X is None or y is None:
            raise ValueError("X or y is None")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X and y don't match: {X.shape[0]} != {y.shape[0]}")
        if X.shape[1] != self.layers[0].input_size:
            raise ValueError(f"Input size of the first layer doesn't match: {X.shape[1]} != {self.layers[0].input_size}")
        if y.shape[1] != self.layers[-1].output_size:
            raise ValueError(f"Output size of the last layer doesn't match: {y.shape[1]} != {self.layers[-1].output_size}")
        if batch_size is not None and batch_size > X.shape[0]:
            raise ValueError(f"Batch size is greater than the number of samples: {batch_size} > {X.shape[0]}")
        
        self.summary(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
        
        for i in range(epochs):
            print("Epoch #", i)
            if batch_size is None:
                # Forward propagation
                self.forward(X)

                # Backpropagation
                self.backward(X, y, learning_rate)

            else:
                for j in range(0, X.shape[0], batch_size):
                    # Split the dataset into batches
                    X_batch = X[j:j + batch_size]
                    y_batch = y[j:j + batch_size]

                    # Forward propagation
                    self.forward(X_batch)

                    # Backpropagation
                    self.backward(X_batch, y_batch, learning_rate)

            # Evaluation
            self.evaluate(X, y, x_test, y_test)

    
    def evaluate(self, X, y, x_test, y_test):
        """
        Evaluation
        Args:
            X (ndarray(m, n))      : Input data
            y (ndarray(m, ))       : Target data
            x_test (ndarray(m, n)) : Input data for testing
            y_test (ndarray(m, ))  : Target data for testing
            y_hat (ndarray(m, ))   : Predicted data
        """
        y_hat = self.forward(X)
        loss = self.loss_cross_entropy(y, y_hat)
        accuracy_train = self.accuracy(y, y_hat)
        if x_test is not None or y_test is not None:
            y_hat_test = self.forward(x_test)
            accuracy_test = self.accuracy(y_test, y_hat_test)
            print(f"Loss: {loss}  Accuracy Train: {accuracy_train} Accuracy Test: {accuracy_test}")
            self.history['accuracyTest'].append(accuracy_test)
        else:
            print(f"Loss: {loss}  Accuracy Train: {accuracy_train}")
        self.history['loss'].append(loss)
        self.history['accuracyTrain'].append(accuracy_train)

    def predict(self, X):
        """
        Prediction
        Args:
            X (ndarray(m, n))      : Input data

        Returns:
            ndarray(m, )           : Predicted data
        """

        # Return the index of the highest probability
        return np.argmax(self.forward(X), axis=1)

    def loss_cross_entropy(self, y, y_pred):
        """
        Cross entropy loss function
        Args:
            y (ndarray(m, ))       : Target data
            y_pred (ndarray(m, ))  : Predicted data

        Returns:
            float                  : Loss 1/N * sum(y * log(y_pred))
        """

        return -np.mean(y * np.log(y_pred + 1e-10))

    def loss_mse(self, y, y_pred):
        """
        Mean squared error loss function
        Args:
            y (ndarray(m, ))       : Target data
            y_pred (ndarray(m, ))  : Predicted data
            
        Returns:
            float                  : Loss 1/N * sum((y - y_pred)^2)
        """
 
        return np.mean(np.square(y - y_pred))
    
    def accuracy(self, y, y_pred):
        """
        Accuracy
        Args:
            y (ndarray(m, ))       : Target data
            y_pred (ndarray(m, ))  : Predicted data

        Returns:
            float                  : Accuracy (Number of correct predictions) / (Total number of predictions)
        """

        return np.sum(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)) / len(y)
    
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

    def summary(self,learning_rate=None, batch_size=None, epochs=None):
        """
        Print a summary of the model
        """

        print("---------------")
        print("Summary:")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Input size: {self.layers[0].input_size}")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}: {layer.input_size} -> {layer.output_size} ({layer.activation_function})")
        print(f"Output size: {self.layers[-1].output_size}")
        print("---------------")