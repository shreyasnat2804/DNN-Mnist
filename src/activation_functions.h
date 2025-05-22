#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath>     // For std::exp, std::tanh
#include <algorithm> // For std::max (used in ReLU)
// #include <vector> // Will be needed if you add Softmax later

// Using double for precision, can be changed to float if preferred for performance/memory.
namespace Activations {

    /**
     * @brief Computes the Sigmoid activation function.
     * S(x) = 1 / (1 + exp(-x))
     * @param x The input value.
     * @return The sigmoid of x (between 0 and 1).
     */
    double sigmoid(double x);

    /**
     * @brief Computes the derivative of the Sigmoid function.
     * S'(x) = S(x) * (1 - S(x))
     * @param activated_output_y The output of the sigmoid function (i.e., sigmoid(x)).
     * @return The derivative of the sigmoid function.
     */
    double sigmoid_derivative(double activated_output_y);

    /**
     * @brief Computes the ReLU (Rectified Linear Unit) activation function.
     * R(x) = max(0, x)
     * @param x The input value.
     * @return The ReLU of x (x if x > 0, else 0).
     */
    double relu(double x);

    /**
     * @brief Computes the derivative of the ReLU function.
     * R'(x) = 1 if x > 0, else 0.
     * @param input_x The original input to the ReLU function.
     * @return The derivative of the ReLU function (1.0 or 0.0).
     */
    double relu_derivative(double input_x);

    /**
     * @brief Computes the Tanh (Hyperbolic Tangent) activation function.
     * T(x) = tanh(x)
     * Using 'tanh_activation' to avoid potential conflicts with std::tanh if not fully namespaced everywhere.
     * @param x The input value.
     * @return The tanh of x (between -1 and 1).
     */
    double tanh_activation(double x);

    /**
     * @brief Computes the derivative of the Tanh function.
     * T'(x) = 1 - tanh(x)^2
     * @param activated_output_y The output of the tanh function (i.e., tanh_activation(x)).
     * @return The derivative of the tanh function.
     */
    double tanh_derivative(double activated_output_y);

    // Note: Softmax is often handled differently as it operates on a vector of scores,
    // typically in the output layer, and its derivative is more complex (a Jacobian matrix)
    // or combined with the cross-entropy loss derivative for simplification.
    // We can add it later if needed, possibly with a different signature.

} // namespace Activations

#endif // ACTIVATION_FUNCTIONS_H