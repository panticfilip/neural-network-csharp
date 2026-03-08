# Neural Network Library (C#)

Simple neural network implementation written from scratch in C#.

This project provides a lightweight neural network library that supports both **classification** and **regression** tasks without relying on external machine learning frameworks.

The goal of this project is to demonstrate how neural networks work internally, including feedforward computation, backpropagation, and weight updates.

---

# Features

* Fully connected feedforward neural network
* ReLU activation for hidden layers
* Softmax output for classification
* Sigmoid output for regression
* Backpropagation implementation
* Mini-batch gradient descent
* Dataset shuffling during training
* Accuracy tracking during training
* Saving and loading network weights

---

# Project Structure

### Sample

Represents a single training example.

```
Sample
- input[]   -> input vector
- output[]  -> expected output
```

---

### Neuron

Represents a single neuron in the network.

Contains:

* weights
* bias
* activation value
* gradients for backpropagation

---

### NeuralNetwork

Base class implementing the core neural network functionality.

Main responsibilities:

* network initialization
* feedforward computation
* backpropagation
* weight updates
* training loop
* saving/loading weights

---

### ClassificationNetwork

Extends `NeuralNetwork` and adds functionality for **classification tasks**.

Features:

* Softmax output layer
* predicted class selection
* accuracy tracking

Main methods:

```
Prediction()
TestNetwork(input)
```

---

### RegressionNetwork

Extends `NeuralNetwork` for **regression problems**.

Features:

* Sigmoid output activation
* tolerance-based correctness check
* vector prediction output

Main methods:

```
Prediction()
TestNetwork(input)
```

---

# Training Process

Training follows these steps:

1. Load existing weights (if available)
2. Shuffle training samples
3. Perform feedforward pass
4. Compute gradients using backpropagation
5. Update weights using mini-batch gradient descent
6. Track accuracy per epoch

---

# Saving and Loading Weights

Weights and biases can be saved to a file using:

```
SaveWeights()
```

and loaded with:

```
LoadWeights()
```

If the file structure does not match the network architecture, an exception is thrown.

---

# Example Usage

```
int[] layers = { 8, 8, 3 };

ClassificationNetwork net = new ClassificationNetwork(
    layers,
    inputSize: 4,
    lr: 0.01,
    batchSz: 16,
    ep: 100
);

net.AddPath("weights.txt");

net.Train(samples, true);

int prediction = net.TestNetwork(input);
```

---

# Purpose of the Project

This library was built primarily as an educational implementation to understand:

* how neural networks work internally
* how backpropagation updates weights
* how training loops are structured

The implementation avoids external ML frameworks in order to keep the learning process transparent.

---

# Author

Filip Pantić
