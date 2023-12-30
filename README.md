# FNN (Feedforward Neural Network) Implementation

This repository contains a basic Python implementation of a Feedforward Neural Network (FNN), designed for both educational and practical purposes. The FNN is a type of artificial neural network where information moves in only one direction—forward—from the input layer, through the hidden layers, and finally to the output layer.

## Files

### FNN.py

The `FNN` class in this module represents the implementation of the Feedforward Neural Network. Key functionalities include:

- **Initialization**: Initialize the network with specified input, hidden, and output layer sizes. Weights are initialized based on the chosen method: random, He, or Glorot.

- **Activation Functions**: Sigmoid activation for the hidden layer and softmax activation for the output layer.

- **Forward Propagation**: Calculate the output of the network given an input.

- **Backward Propagation**: Update weights and biases based on the calculated error to minimize the loss.

- **Training**: Train the network using provided training data, adjusting weights through epochs.

- **Evaluation and Prediction**: Evaluate the accuracy of the model on test data and make predictions.

- **Save and Load Weights**: Functions to save and load the trained weights and biases.

- **Loss Functions**: Cross-entropy and Mean Squared Error (MSE) loss functions.

### Data.py

This module focuses on fetching and preprocessing the MNIST dataset. The dataset is divided into training and testing sets, normalized to the range [0, 1], and saved as NumPy arrays.

### train.py

In this module, the FNN is trained using the MNIST dataset. The training data is loaded, one-hot encoded, and the model is trained for a specified number of epochs. The accuracy of the model on the test set is evaluated, and predictions are made on a sample of test data.

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
