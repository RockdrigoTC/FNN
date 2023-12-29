import numpy as np
from sklearn.datasets import fetch_openml

# Cargar dataset MNIST
print("Cargando dataset MNIST...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
mnist.target = mnist.target.astype(np.int8)

# Dividir el dataset en conjuntos de entrenamiento y prueba
print("Dividiendo el dataset en conjuntos de entrenamiento y prueba...")
train_size = 60000
X_train, X_test, y_train, y_test = mnist.data[:train_size], mnist.data[train_size:], mnist.target[:train_size], mnist.target[train_size:]

# Normalizar las imágenes para que los valores estén en el rango [0, 1]
print("Normalizando las imágenes...")
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Información del dataset:")
print(f'len(X_train): {len(X_train)} shape: {X_train.shape}')
print(f'len(X_test): {len(X_test)} shape: {X_test.shape}')
print(f'len(y_train): {len(y_train)} shape: {y_train.shape}')
print(f'len(y_test): {len(y_test)} shape: {y_test.shape}')

# Guardar los datos en archivos numpy
print("Guardando los datos en archivos numpy...")
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

