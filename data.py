import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Cargar dataset MNIST
mnist = fetch_openml('mnist_784', version=1, parser='auto')
mnist.target = mnist.target.astype(np.int8)

# Dividir el dataset en conjuntos de entrenamiento y prueba
train_size = 60000
X_train, X_test, y_train, y_test = mnist.data[:train_size], mnist.data[train_size:], mnist.target[:train_size], mnist.target[train_size:]

# Normalizar las imágenes para que los valores estén en el rango [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

# Guardar los datos en archivos numpy
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)


""" # Transponer las matrices de imágenes
X_train = X_train.T
X_test = X_test.T """

# Guardar los datos en archivos CSV
np.savetxt("X_train.csv", X_train, delimiter=',', fmt='%f')
np.savetxt("X_test.csv", X_test, delimiter=',', fmt='%f')
np.savetxt("y_train.csv", y_train, delimiter=',', fmt='%d')
np.savetxt("y_test.csv", y_test, delimiter=',', fmt='%d')
