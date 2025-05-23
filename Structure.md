---
## Modular C++ DNN Project Structure Report

This report outlines a highly modular C++ project structure for building a Deep Neural Network (DNN), specifically targeting MNIST character recognition with a goal of 90% accuracy. This design emphasizes separation of concerns, reusability, and maintainability, which are crucial for a complex application and an excellent learning approach for C++.

---
### Core Modules

The project is organized into distinct modules, each handling a specific functional aspect of the DNN:

1.  **`mnist_reader.h`/`.cpp`**:
    * **Purpose:** Dedicated to the initial data ingress. This module is responsible for accurately reading and parsing the raw MNIST binary files (images and labels) from disk.
    * **Key Functionality:** Handles file I/O and understanding the specific IDX file format of the MNIST dataset.

2.  **`data_processor.h`/`.cpp`**:
    * **Purpose:** Manages the Extract, Transform, and Load (ETL) pipeline for the MNIST data.
    * **Key Functionality:** Takes the raw data from `mnist_reader`, normalizes pixel values (e.g., from 0-255 to 0.0-1.0), and reshapes image data into a format suitable for network input (e.g., flattened 1D vectors).

3.  **`activation_functions.h`/`.cpp`**:
    * **Purpose:** Encapsulates various activation functions and their corresponding derivatives, essential for neural network computations.
    * **Key Functionality:** Provides implementations for functions like Sigmoid, ReLU, Tanh, and Softmax, along with their mathematical derivatives required for backpropagation.

4.  **`loss_functions.h`/`.cpp`**:
    * **Purpose:** Defines the objective functions used to measure the network's performance during training.
    * **Key Functionality:** Implements common loss functions, such as Cross-Entropy Loss, and their derivatives, which guide the learning process.

5.  **`neuron.h`/`.cpp`**:
    * **Purpose:** The fundamental building block of the neural network.
    * **Key Functionality:** Defines the properties and behavior of a single neuron, including its weights, bias, and how it computes its output using an activation function.

6.  **`layer.h`/`.cpp`**:
    * **Purpose:** Represents a collection of neurons forming a distinct layer within the network.
    * **Key Functionality:** Manages the forward pass (computing outputs for all neurons in the layer) and the backward pass (propagating gradients back through the layer).

7.  **`optimizer.h`/`.cpp`**:
    * **Purpose:** Implements the algorithms responsible for updating the network's weights and biases during training.
    * **Key Functionality:** Provides different optimization strategies, such as Stochastic Gradient Descent (SGD), which utilize gradients to minimize the loss function.

8.  **`neural_network.h`/`.cpp`**:
    * **Purpose:** The central orchestrator, bringing together all the layers to form the complete DNN.
    * **Key Functionality:** Manages the overall network structure, handles the end-to-end forward and backward propagation, and controls the training and prediction processes using the specified optimizer and loss function.

9.  **`main.cpp`**:
    * **Purpose:** The application's entry point.
    * **Key Functionality:** Drives the entire workflow: loads and processes data, initializes the neural network, configures training parameters (e.g., learning rate, epochs), initiates training, and evaluates the model's performance on test data.

---