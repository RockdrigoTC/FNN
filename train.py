import numpy as np
from FNN import FNN

# Load data from numpy files
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Encode labels in one-hot format
num_classes = 10
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]
    
input_size = X_train.shape[1]
hidden_size = 100
epoch = 100
output_size = num_classes

# Create and train the model
model = FNN(input_size, hidden_size, output_size, weigth_init='he') # weigth_init=('he' or 'glorot' or 'random)
model.train(X_train, y_train_one_hot, X_test, y_test_one_hot , epochs=epoch, learning_rate=0.9)
model.saveWeightsBiases()

# Load saved weights and biases
model.loadWeightsBiases()

# Predictions
sample_indices = np.random.choice(len(X_test), size=10, replace=False)
for i in sample_indices:
    predict = model.predict(X_test[i].reshape(1, -1))
    predicted_class = np.argmax(predict)
    real_class = y_test[i]
    print(f"Predicted class: {predicted_class} ({predict.astype(int)})")
    print(f"Real class: {real_class}")
    print("-----------")