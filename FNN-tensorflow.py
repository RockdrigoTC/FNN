import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# FNN tensorflow para clasificacion de imagenes
# Dataset: MNIST

# Cargar dataset
mnist = tf.keras.datasets.mnist

# Carga el conjunto de datos en dos conjuntos: entrenamiento y prueba
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normaliza las imágenes para que los valores estén en el rango [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Aplana las imágenes
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# Codifica las etiquetas en formato one-hot
train_labels_one_shot = tf.keras.utils.to_categorical(train_labels, 10)

# sin uso de GPU


# Crea el modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compila el modelo
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Detener el entrenamiento cuando la perdida no mejora sustancialmente
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.1):
            print("\nLoss is low so cancelling training!")
            self.model.stop_training = True


# Entrena el modelo con los datos de entrenamiento aplanados.
model.fit(train_images, train_labels_one_shot, epochs=1000)

# Evalua el modelo con los datos de prueba aplanados.
model.evaluate(test_images, tf.keras.utils.to_categorical(test_labels, 10))

# Predice la clase de una imagen
predictions = model.predict(test_images)
print(np.argmax(predictions[0]))
print(test_labels[0])

# Guarda el modelo
model.save("model.h5")

# Carga el modelo
model = tf.keras.models.load_model("model.h5")

# Predice la clase de una imagen
predictions = model.predict(test_images)
print(np.argmax(predictions[3]))
print(test_labels[3])


