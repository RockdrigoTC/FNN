import numpy as np
from FNN import FNN

# Cargar los datos desde los archivos numpy
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Codificar las etiquetas en formato one-hot
num_classes = 10
y_train_one_hot = np.eye(num_classes)[y_train]
    
input_size = X_train.shape[1]
hidden_size = 100
epoch = 100
output_size = num_classes

# Crear y entrenar el modelo
model = FNN(input_size, hidden_size, output_size, weigth_init='random')
model.train(X_train, y_train_one_hot, epochs=epoch, learning_rate=0.9)
model.saveWeightsBiases()

# Cargar los pesos y sesgos guardados
model.loadWeightsBiases()

# Evaluar el modelo
y_test_one_hot = np.eye(num_classes)[y_test]
accuracy = model.evaluate(X_test, y_test_one_hot)
print("Accuracy:", accuracy)

# Predecir
for i in range(10):
    predict = model.predict(X_test[i].reshape(1, -1))
    predicted_class = np.argmax(predict)
    real_class = y_test[i]
    print(f"Clase predicha: {predicted_class} ({predict.astype(int)})")
    print(f"Clase real: {real_class}")
    print("-----------")