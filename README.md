# FNN (Feedforward Neural Network) Implementation

This repository contains a basic Python implementation of a Feedforward Neural Network (FNN), designed for both educational and practical purposes. The FNN is a type of artificial neural network where information moves in only one direction—forward—from the input layer, through the hidden layers, and finally to the output layer.

## Files

### FNN.py

The `FNN` class in this module represents the implementation of the Feedforward Neural Network. Key functionalities include:

- **Initialization**: Initialize the network with specified input, hidden, and output layer sizes. Weights are initialized based on the chosen method: random, He, or Glorot.

- **Activation Functions**: Sigmoid, Tanh, ReLU, and Leaky ReLU activation functions.

- **Optimizers**: Stochastic Gradient Descent (SGD), SGD with Momentum, RMSProp, and Adam optimizers.

- **Forward Propagation**: Calculate the output of the network given an input.

- **Backward Propagation**: Update weights and biases based on the calculated error to minimize the loss.

- **Training**: Train the network using provided training data, adjusting weights through epochs.

- **Evaluation and Prediction**: Evaluate the accuracy of the model on test data and make predictions.

- **Save and Load**: Save and load the model to and from a file.

- **Loss Functions**: Cross-entropy and Mean Squared Error (MSE) loss functions.

### data.py

This module contains functions to download and prepare datasets for training. Some of the datasets include: MNIST Handwritten Digits, Fashion-MNIST, Wine Quality, Diabetes, and Iris. You can also use other datasets. But probably you need to modify the code a little bit.

### train.py

In this module, you can train the FNN on the dataset you want. You can also change the hyperparameters of the model, such as the number of hidden layers, the activation function, the optimizer, the learning rate, the batch size, the number of epochs, and so on.

## Instructions

1. Ensure you have the required dependencies installed:

   ```bash
   pip install numpy scikit-learn

2. Run the 'data.py' file to download and prepare the data(MNIST dataset) for training:

   ```bash
   python data.py

3. Run the train.py script to train the FNN on the MNIST dataset:

   ```bash
   python data.py

(Optional) Install matplotlib to display training statistics plots

   ```bash
   pip install matplotlib
