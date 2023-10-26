import numpy as np


# Crear el modelo
class FNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicializar pesos
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
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

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for i in range(epochs):
            print("Epoch #", i)
            y_pred = self.forward(X)
            self.backward(X, y, learning_rate)
            print("Loss:", self.loss_cross_entropy(y, y_pred))

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return accuracy
    
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
    


# Cargar dataset MNIST
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(np.int8)

# Dividir el dataset en conjuntos de entrenamiento y prueba
train_size = 60000
X_train, X_test, y_train, y_test = mnist.data[:train_size], mnist.data[train_size:], mnist.target[:train_size], mnist.target[train_size:]

# Normalizar las imágenes para que los valores estén en el rango [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Codificar las etiquetas en formato one-hot
num_classes = 10
y_train_one_hot = np.eye(num_classes)[y_train]
    

# Crear y entrenar el modelo
input_size = X_train.shape[1]
hidden_size = 100
output_size = num_classes
model = FNN(input_size, hidden_size, output_size)
model.train(X_train, y_train_one_hot, epochs=50, learning_rate=0.01)

# Evaluar el modelo
y_test_one_hot = np.eye(num_classes)[y_test]
accuracy = model.evaluate(X_test, y_test_one_hot)
print("Accuracy:", accuracy)