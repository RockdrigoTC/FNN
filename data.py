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
""" np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test) """


# Crear DataFrames de pandas
""" df_X_train = pd.DataFrame(X_train)
df_X_test = pd.DataFrame(X_test) """
df_y_train = pd.DataFrame(y_train)
df_y_test = pd.DataFrame(y_test)

# Guardar DataFrames en archivos CSV
""" df_X_train.to_csv("X_train.csv", index=False)
df_X_test.to_csv("X_test.csv", index=False) """
df_y_train.to_csv("y_train.csv", index=False)
df_y_test.to_csv("y_test.csv", index=False)
