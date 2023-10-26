# FNN(Feedforward Neural Network) implementation

import numpy as np

class FNN:
    def __init__(self, input_size, hidden_size, output_size, weigth_init="random"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if weigth_init == 'he':
            # Inicialización de pesos con He
            scale = np.sqrt(2.0 / input_size)
        elif weigth_init == 'glorot':
            # Inicialización de pesos con Glorot
            scale = np.sqrt(2.0 / (input_size + output_size))
        elif weigth_init == 'random':
            scale = 1.0
        else:
            raise ValueError("Tipo de inicializacion no reconocida.")
            
        # Inicializar pesos
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * scale
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * scale
        self.b2 = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]

        self.error = self.a2 - y

        self.grad_W2 = np.dot(self.a1.T, self.error) / m
        self.grad_b2 = np.sum(self.error, axis=0) / m

        self.error2 = np.dot(self.error, self.W2.T) * (self.a1 * (1 - self.a1))

        self.grad_W1 = np.dot(X.T, self.error2) / m
        self.grad_b1 = np.sum(self.error2, axis=0) / m

        self.W2 -= learning_rate * self.grad_W2
        self.b2 -= learning_rate * self.grad_b2
        self.W1 -= learning_rate * self.grad_W1
        self.b1 -= learning_rate * self.grad_b1

    def train(self, X, y, epochs=10, learning_rate=0.01):
        for i in range(epochs):
            print("Epoch #", i)
            y_pred = self.forward(X)
            self.backward(X, y, learning_rate)
            print(f"Loss: {self.loss_cross_entropy(y, y_pred)}")

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return accuracy

    def predict(self, X):
      return np.round(self.forward(X))

    def saveWeightsBiases(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")
        np.savetxt("b1.txt", self.b1, fmt="%s")
        np.savetxt("b2.txt", self.b2, fmt="%s")


    def loadWeights(self):
        self.W1 = np.loadtxt("w1.txt", dtype=float)
        self.W2 = np.loadtxt("w2.txt", dtype=float)
        self.b1 = np.loadtxt("b1.txt", dtype=float)
        self.b2 = np.loadtxt("b2.txt", dtype=float)
    
    # Funcion de perdida(cross-entropy)
    def loss_cross_entropy(self, y, y_pred):
        return -np.mean(y * np.log(y_pred + 1e-10))
    
    # Funcion de perdida(mse)
    def loss_mse(self, y, y_pred):
        return np.mean(np.square(y - y_pred))
    
    def loss(self, y, y_pred):
        return self.loss_cross_entropy(y, y_pred)
    
    def loss_categorical_crossentropy(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-10))
    

