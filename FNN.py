# FNN(Feedforward Neural Network) implementation

import numpy as np
#import tensorflow as tf

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
        self.z2 = self.softmax(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.softmax(self.z3)
        return o

    def softmax(self, X):
        # Funcion de activacion softmax
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)
    
    def softmax_derivative(self, X):
        # Derivada de la funcion de activacion softmax
        s = self.softmax(X)
        return s * (1 - s)

    def sigmoid(self, s):
        # Funcion de activacion
        return 1 / (1 + np.exp(-s))
    
    def sigmoidPrime(self, s):
        # Derivada de la funcion de activacion
        return s * (1 - s)
    
    def backward(self, X, y, o, learning_rate=0.1):
        # Propagacion hacia atras
        self.o_error = y - o
        self.o_delta = self.o_error * self.softmax_derivative(o)
        
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.softmax_derivative(self.z2)
        
        self.W1 += X.T.dot(self.z2_delta) * learning_rate
        self.W2 += self.z2.T.dot(self.o_delta) * learning_rate
        
    def train(self, X, y, learning_rate=0.1):
        o = self.forward(X)
        self.backward(X, y, o, learning_rate)
        
    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def loadWeights(self):
        self.W1 = np.loadtxt("w1.txt", dtype=float)
        self.W2 = np.loadtxt("w2.txt", dtype=float)
        
    def predict(self, X):
        print("Resultado: \n" + str(np.round(self.forward(X))))

    def loss(self, X, y):
        o = self.forward(X)
        return np.mean(np.square(y - o))
    
    def categorical_crossentropy(self, y_true, y_pred):
        # Convierte y_true en formato one-hot
        num_classes = y_pred.shape[1]
        y_true_onehot = np.eye(num_classes)[y_true]

        # Calcula la Entropía Cruzada Categórica
        epsilon = 1e-15
        loss = -np.sum(y_true_onehot * np.log(y_pred + epsilon)) / len(y_true)

        return loss
    
def one_shot_encode(labels, num_classes):
    # Crea una matriz de ceros con la forma (len(labels), num_classes)
    encoded_labels = np.zeros((len(labels), num_classes))
    
    # Establece el valor correspondiente a 1 en la columna de la clase adecuada
    for i in range(len(labels)):
        encoded_labels[i, labels[i]] = 1
    
    return encoded_labels


if __name__ == "__main__":

    # # Descarga el conjunto de datos MNIST
    # mnist = tf.keras.datasets.mnist

    # # Carga el conjunto de datos en dos conjuntos: entrenamiento y prueba
    # (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # # Normaliza las imágenes para que los valores estén en el rango [0, 1]
    # train_images, test_images = train_images / 255.0, test_images / 255.0

    # # Guarda los datos de entrenamiento y prueba en archivos numpy
    # np.save("train_images.npy", train_images)
    # np.save("train_labels.npy", train_labels)
    # np.save("test_images.npy", test_images)
    # np.save("test_labels.npy", test_labels)

    # Carga los datos de entrenamiento y prueba desde los archivos numpy
    train_images = np.load("train_images.npy")
    train_labels = np.load("train_labels.npy")
    test_images = np.load("test_images.npy")
    test_labels = np.load("test_labels.npy")

    # Aplana las imágenes
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    # Codifica las etiquetas en formato one-hot
    train_labels_one_shot = one_shot_encode(train_labels, 10)

    # Crea el modelo
    model = FNN(784, 100, 10)

    epoch = 50

    # Entrena el modelo con los datos de entrenamiento aplanados.
    for i in range(epoch):
        print("Epoch: " + str(i))
        model.train(train_images, train_labels_one_shot, learning_rate=0.9)
        print("Loss: " + str(model.categorical_crossentropy(train_labels, model.forward(train_images))))

    # Guarda los pesos del modelo
    model.saveWeights()

    # Carga los pesos del modelo
    model.loadWeights()

    # Codifica las etiquetas de prueba en formato one-hot
    test_labels = one_shot_encode(test_labels, 10)

    # Evalua el modelo con los datos de prueba aplanados.
    print("Loss: " + str(model.loss(test_images, test_labels)))

    # Predice la clase de una imagen
    model.predict(test_images[0].reshape(1, -1))
    print("Clase real: " + str(test_labels[0]))


