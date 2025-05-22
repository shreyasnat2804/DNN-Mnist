#include "activation_functions.h"

#include <cmath>       // For std::exp, std::tanh (already in .h but good for clarity)
#include <algorithm>   // For std::max (already in .h but good for clarity)
#include <stdexcept>   // For potential error handling if needed in more complex functions

namespace Activations {

    // --- Sigmoid ---
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoid_derivative(double activated_output_y) {
        // Assumes activated_output_y is sigmoid(x)
        // S'(x) = S(x) * (1 - S(x))
        return activated_output_y * (1.0 - activated_output_y);
    }

    // --- ReLU (Rectified Linear Unit) ---
    double relu(double x) {
        return std::max(0.0, x);
    }

    double relu_derivative(double input_x) {
        // R'(x) = 1 if x > 0, else 0
        return (input_x > 0.0) ? 1.0 : 0.0;
    }

    // --- Tanh (Hyperbolic Tangent) ---
    double tanh_activation(double x) {
        return std::tanh(x);
    }

    double tanh_derivative(double activated_output_y) {
        // Assumes activated_output_y is tanh_activation(x)
        // T'(x) = 1 - (tanh(x))^2
        return 1.0 - (activated_output_y * activated_output_y);
    }

} // namespace Activations