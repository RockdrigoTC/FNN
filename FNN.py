# FNN(Feedforward Neural Network) implementation

import numpy as np

class FNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Definir hiperpametros
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicializar pesos
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        
    def forward(self, X):
        # Propagacion hacia adelante
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o
    
    def sigmoid(self, s):
        # Funcion de activacion
        return 1 / (1 + np.exp(-s))
    
    def sigmoidPrime(self, s):
        # Derivada de la funcion de activacion
        return s * (1 - s)
    
    def backward(self, X, y, o):
        # Propagacion hacia atras
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)
        
    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
        
    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def loadWeights(self):
        self.W1 = np.loadtxt("w1.txt", dtype=float)
        self.W2 = np.loadtxt("w2.txt", dtype=float)
        
    def predict(self, X):
        print("Resultado: \n" + str(np.round(self.forward(X))))


if __name__ == "__main__":

    # Carga los datos
    train_data = np.load('train_X.npy')
    train_label = np.load('train_label.npy')
    valid_data = np.load('valid_X.npy')
    valid_label = np.load('valid_label.npy')
    test_data = np.load('test_X.npy')
    test_label = np.load('test_Y.npy')

    # train_data.shape = (num_examples, 21, 28, 3)
    # train_label.shape = (num_examples, 10)
    # valid_data.shape = (num_examples, 21, 28, 3)
    # valid_label.shape = (num_examples, 10)



    # Definir hiperparametros
    input_size = train_data.shape[1] * train_data.shape[2] * train_data.shape[3]
    hidden_size = 100
    output_size = train_label.shape[1]
    learning_rate = 0.1
    epochs = 1000

    train_data = train_data.reshape(train_data.shape[0], -1)
    valid_data = valid_data.reshape(valid_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)


    # Crear red neuronal
    nn = FNN(input_size, hidden_size, output_size)

    # Entrenar red neuronal
    for i in range(epochs):
        print("Epoch: " + str(i+1))
        if i % 100 == 0:
            print("Loss: \n" + str(np.mean(np.square(valid_label - nn.forward(valid_data)))))
        nn.train(train_data, train_label)

    # Guardar pesos
    nn.saveWeights()

    # Cargar pesos
    nn.loadWeights()

    # Predecir
    nn.predict(test_data[5])
    nn.predict(test_data[8])

    print("Test data: \n" + str(test_label[5]))
    print("Test data: \n" + str(test_label[8]))

    