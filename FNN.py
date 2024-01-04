import numpy as np
import pickle

# Layer class definition
class Layer:
    """
    Layer(obj)

    ----------
    Parameters
    ----------
        input_size (int)              : Size of the input layer
        output_size (int)             : Size of the output layer
        activation_function (string)  : Activation function
        optimizer (string)            : Optimizer method
        weight_init (string)          : Weight initialization method
        beta1 (float)                 : Beta 1 (Used in momentum, RMSprop and Adam optimizers)
        beta2 (float)                 : Beta 2 (Used in Adam optimizer)
        dropout (float)               : Dropout rate
    """
    def __init__(self, input_size=None, output_size=None, activation='sigmoid',optimizer='sgd', weight_init='random', beta1=0.9, beta2=0.999, dropout=0.0):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation
        self.optimizer = optimizer
        self.dropout_rate = dropout
        self.dropout_mask = None
        self.error = None
        self.t = 0

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

        # Optimizer moments initialization
        self.v_weights = np.zeros_like(self.weights)
        self.v_biases = np.zeros_like(self.biases)
        self.s_weights = np.zeros_like(self.weights)
        self.s_biases = np.zeros_like(self.biases)
        self.beta1 = beta1
        self.beta2 = beta2

        # Adagrad
        self.G_weights = np.zeros_like(self.weights)
        self.G_biases = np.zeros_like(self.biases)

    def activate(self, x):
        """
        Description: Compute the activation function for the input data, and apply Dropout if necessary

        ----------
        Parameters
        ----------
            x (ndarray)  : Input data

        Returns
        -------
            ndarray      : Result of the activation function
        """
        if self.dropout_rate > 0.0:
            # Generar una máscara de Dropout y aplicarla
            self.dropout_mask = (np.random.rand(*x.shape) >= self.dropout_rate).astype(float)
            x *= self.dropout_mask
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
        
    def update_weights(self, learning_rate, grad_weights, grad_biases):
        """
        Update weights and biases using the specified optimizer

        ----------
        Parameters
        ----------
            learning_rate (float)   : Learning rate (alpha)
            grad_weights (ndarray)  : Gradient of weights
            grad_biases (ndarray)   : Gradient of biases
        
        Returns
        -------
            None
        """
        if self.optimizer == 'sgd':
            # Stochastic gradient descent
            self.weights -= learning_rate * grad_weights
            self.biases -= learning_rate * grad_biases
        elif self.optimizer == 'momentum':
            # Momentum
            self.v_weights = self.beta1 * self.v_weights + (1 - self.beta1) * grad_weights
            self.v_biases = self.beta1 * self.v_biases + (1 - self.beta1) * grad_biases
            self.weights -= learning_rate * self.v_weights
            self.biases -= learning_rate * self.v_biases
        elif self.optimizer == 'rmsprop':
            # RMSprop
            self.s_weights = self.beta1 * self.s_weights + (1 - self.beta1) * np.square(grad_weights)
            self.s_biases = self.beta1 * self.s_biases + (1 - self.beta1) * np.square(grad_biases)
            self.weights -= learning_rate * grad_weights / (np.sqrt(self.s_weights) + 1e-10)
            self.biases -= learning_rate * grad_biases / (np.sqrt(self.s_biases) + 1e-10)
        elif self.optimizer == 'adam':
            # Adam
            self.v_weights = self.beta1 * self.v_weights + (1 - self.beta1) * grad_weights
            self.v_biases = self.beta1 * self.v_biases + (1 - self.beta1) * grad_biases
            self.s_weights = self.beta2 * self.s_weights + (1 - self.beta2) * np.square(grad_weights)
            self.s_biases = self.beta2 * self.s_biases + (1 - self.beta2) * np.square(grad_biases)
            self.t += 1
            v_weights_corrected = self.v_weights / (1 - self.beta1 ** self.t)
            v_biases_corrected = self.v_biases / (1 - self.beta1 ** self.t)
            s_weights_corrected = self.s_weights / (1 - self.beta2 ** self.t)
            s_biases_corrected = self.s_biases / (1 - self.beta2 ** self.t)
            self.weights -= learning_rate * v_weights_corrected / (np.sqrt(s_weights_corrected) + 1e-10)
            self.biases -= learning_rate * v_biases_corrected / (np.sqrt(s_biases_corrected) + 1e-10)
        elif self.optimizer == 'adagrad':
            # Adagrad
            self.G_weights += np.square(grad_weights)
            self.G_biases += np.square(grad_biases)
            self.weights -= learning_rate * grad_weights / (np.sqrt(self.G_weights) + 1e-10)
            self.biases -= learning_rate * grad_biases / (np.sqrt(self.G_biases) + 1e-10)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
            
# Feedforward Neural Network class definition
class FNN:
    """
    FNN(obj)

    ----------
    Parameters
    ----------
        layers (list)  : List of Layer objects
    """
    def __init__(self, layers=None):
        self.layers = layers
        self.history = {'loss': [], 'accuracyTrain': [], 'accuracyTest': []}

    def forward(self, X):
        """
        Forward propagation through the network

        Parameters
        ----------
            X (ndarray)            : Input data

        Returns
        -------
            activations (ndarray)  : Output data
        """
        activations = X
        for layer in self.layers:
            z = np.dot(activations, layer.weights) + layer.biases
            activations = layer.activate(z)
        return activations

    def backward(self, X, y, learning_rate=0.01):
        """
        Backward propagation through the network. Compute gradients for each layer and update weights and biases using the specified optimizer.

        ----------
        Parameters
        ----------
            X (ndarray)            : Input data
            y (ndarray)            : Target data
            learning_rate (float)  : Learning rate (alpha)
        
        Returns
        -------
            None
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

            # Apply Dropout during backpropagation
            if layer.dropout_rate > 0.0:
                layer.error *= layer.dropout_mask

            # Compute gradients for the layer
            layer.grad_weights = np.dot(prev_output.T, layer.error) / m
            layer.grad_biases = np.sum(layer.error, axis=0) / m

            # Update weights and biases using the specified optimizer
            layer.update_weights(learning_rate, layer.grad_weights, layer.grad_biases)

            # Store the error for the next iteration
            next_layer = layer

    def train(self, X=None, y=None, x_test=None, y_test=None, epochs=10, learning_rate=0.01, batch_size=None, patience=0):
        """
        Train the model and evaluate it after each epoch

        ----------
        Parameters
        ----------
            X (ndarray)            : Input data
            y (ndarray)            : Target data
            x_test (ndarray)       : Input data for testing (Optional)
            y_test (ndarray)       : Target data for testing (Optional)
            epochs (int)           : Number of epochs (Number of times the entire dataset is passed forward and backward through the neural network)
            learning_rate (float)  : Learning rate (alpha)
            batch_size (int)       : Batch size (Number of samples per gradient update)
            patience (int)         : Patience for early stopping (Number of epochs with no improvement after which training will be stopped)

        Returns
        -------
            None
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
        
        self.summary(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, patience=patience)
        
        best_accuracy = 0.0
        wait = 0

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

            # Early stopping
            stop, best_accuracy, wait = self.early_stopping(x_test, y_test, patience, i, best_accuracy, wait)
            if stop: break

    def evaluate(self, X, y, x_test, y_test):
        """
        Evaluate the model and print the loss and accuracy for the training and testing sets

        ----------
        Parameters
        ----------
            X (ndarray)       : Input data
            y (ndarray)       : Target data
            x_test (ndarray)  : Input data for testing
            y_test (ndarray)  : Target data for testing
            y_hat (ndarray)   : Predicted data

        Returns
        -------
            None
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
        Predict the class of the input data using the trained model

        ----------
        Parameters
        ----------
            X (ndarray)  : Input data

        Returns
        -------
            (ndarray)    : Predicted data
        """
        # Return the index of the highest probability
        return np.argmax(self.forward(X), axis=1)

    def loss_cross_entropy(self, y, y_pred):
        """
        Cross entropy loss function

        ----------
        Parameters
        ----------
            y (ndarray)       : Target data
            y_pred (ndarray)  : Predicted data

        Returns
        -------
            (float)           : Loss 1/N * sum(y * log(y_pred))
        """
        return -np.mean(y * np.log(y_pred + 1e-10))

    def loss_mse(self, y, y_pred):
        """
        Mean squared error loss function

        ----------
        Parameters
            y (ndarray)       : Target data
            y_pred (ndarray)  : Predicted data
            
        Returns
            (float)           : Loss 1/N * sum((y - y_pred)^2)
        """
        return np.mean(np.square(y - y_pred))
    
    def loss_hinge(self, y, y_pred):
        """
        Hinge loss function

        ----------
        Parameters
        ----------
            y (ndarray)       : Target data
            y_pred (ndarray)  : Predicted data
            
        Returns
        -------
            (float)           : Loss 1/N * sum(max(0, 1 - y * y_pred))
        """
        return np.mean(np.maximum(0, 1 - y * y_pred))
    
    def accuracy(self, y, y_pred):
        """
        Accuracy metric

        ----------
        Parameters
        ----------
            y (ndarray)       : Target data
            y_pred (ndarray)  : Predicted data

        Returns
        -------
            (float)           : Accuracy (Number of correct predictions) / (Total number of predictions)
        """
        return np.sum(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)) / len(y)  
    
    # Save the model to a file
    def save_model(self, filename):
        """
        Save the model to a file

        ----------
        Parameters
        ----------
            filename (string)  : File name

        Returns
        -------
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    # Load the model from a file
    def load_model(self, filename):
        """
        Load the model from a file

        ----------
        Parameters
        ----------
            filename (string)  : File name
        
        Returns
        -------
            None
        """
        with open(filename, 'rb') as file:
            loaded_model = pickle.load(file)
        # Copy the loaded model's parameters to the current model
        self.__dict__.update(loaded_model.__dict__)

    def early_stopping(self, x_test, y_test, patience, i, best_accuracy, wait):
        """
        Early stopping

        ----------
        Parameters
        -----------
            x_test (ndarray)    : Input data for testing
            y_test (ndarray)    : Target data for testing
            patience (int)      : Patience for early stopping
            i (int)             : Current epoch
            best_accuracy (int) : Best accuracy
            wait (int)          : Number of epochs with no improvement

        Returns
        --------
            stop (bool)         : Early stopping
            best_accuracy (int) : Best accuracy
            wait (int)          : Number of epochs with no improvement
        """
        stop = False
        if patience > 0 and x_test is not None and y_test is not None:
            if self.history['accuracyTest'][-1] > best_accuracy:
                best_accuracy = self.history['accuracyTest'][-1]
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"\nEarly stopping at epoch {i}")
                    stop = True
        return stop, best_accuracy, wait

    def summary(self,learning_rate=None, batch_size=None, epochs=None, patience=None):
        """
        Print a summary of the model

        ----------
        Parameters
        ----------
            learning_rate (float)  : Learning rate
            batch_size (int)       : Batch size
            epochs (int)           : Number of epochs
            patience (int)         : Patience for early stopping

        Returns
        -------
            None
        """
        print("---------------")
        print("**Summary**:\n")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Early stopping(patience): {patience}")
        print(f"Number of layers: {len(self.layers)}")    
        print(f"Input size: {self.layers[0].input_size}")
        print(f"Output size: {self.layers[-1].output_size}")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}: \n  - {layer.input_size} -> {layer.output_size} \n  - Activation: {layer.activation_function}") 
            print(f"  - Optimizer: {layer.optimizer} \n  - Beta 1: {layer.beta1} \n  - Beta 2: {layer.beta2} \n  - Dropout: {layer.dropout_rate}")      
        print("---------------")