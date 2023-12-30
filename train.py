import numpy as np
import matplotlib.pyplot as plt
from FNN import Layer, FNN

# Load data from numpy files
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Encode labels in one-hot format
num_classes = len(np.unique(y_train))
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]
    
input_size = X_train.shape[1]
hidden_size = 100
epoch = 100
output_size = num_classes

# Crear capas
input_layer = Layer(input_size, hidden_size, activation_function='sigmoid', weight_init='he')
hidden_layer = Layer(hidden_size, output_size, activation_function='softmax', weight_init='he') 

# Create and train the model
model = FNN(layers=[input_layer, hidden_layer])
model.train(X=X_train, y=y_train_one_hot, x_test=X_test, y_test=y_test_one_hot , epochs=epoch, learning_rate=0.3)

# Save the model
model.save_model("model.pkl")

# Load the model
model.load_model("model.pkl")

# Predictions
sample_indices = np.random.choice(len(X_test), size=10, replace=False)
for i in sample_indices:
    predict = model.predict(X_test[i].reshape(1, -1))
    real_class = y_test[i]
    print(f"Predicted class: {predict.astype(int)}")
    print(f"Real class: {real_class}")
    print("-----------")

# Plot loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(model.history['loss'])
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.subplot(122)
plt.plot(model.history['accuracyTrain'], label='train')
plt.plot(model.history['accuracyTest'], label='test')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
