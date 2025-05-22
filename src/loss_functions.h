#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <vector>
#include <cmath>       // For std::log, std::pow, std::max
#include <stdexcept>   // For exceptions

// Using double for precision.
namespace LossFunctions {

    // --- Mean Squared Error (MSE) ---

    /**
     * @brief Calculates Mean Squared Error loss.
     * L = (1/D) * sum_d((predictions[d] - targets[d])^2)
     * where D is the number of output elements.
     * @param predictions Vector of predicted values.
     * @param targets Vector of true target values.
     * @return The mean squared error.
     */
    double mean_squared_error(const std::vector<double>& predictions, const std::vector<double>& targets);

    /**
     * @brief Calculates the derivative of MSE loss with respect to predictions.
     * dL/d_prediction[d] = (2/D) * (predictions[d] - targets[d])
     * @param predictions Vector of predicted values.
     * @param targets Vector of true target values.
     * @return Vector of derivatives.
     */
    std::vector<double> mean_squared_error_derivative(const std::vector<double>& predictions, const std::vector<double>& targets);


    // --- Categorical Cross-Entropy Loss ---
    // Typically used with a Softmax output layer.

    /**
     * @brief Calculates Categorical Cross-Entropy loss.
     * Assumes predictions are probabilities (e.g., from Softmax) and targets are one-hot encoded.
     * L = -sum(targets_one_hot[i] * log(predictions[i] + epsilon))
     * @param predictions Vector of predicted probabilities for each class.
     * @param targets_one_hot Vector of true probabilities (one-hot encoded).
     * @param epsilon A small constant to prevent log(0). Defaults to 1e-12.
     * @return The cross-entropy loss value.
     */
    double categorical_cross_entropy(const std::vector<double>& predictions,
                                     const std::vector<double>& targets_one_hot,
                                     double epsilon = 1e-12);

    /**
     * @brief Calculates Categorical Cross-Entropy loss when the true label is an index.
     * Assumes predictions are probabilities (e.g., from Softmax).
     * L = -log(predictions[true_class_index] + epsilon)
     * @param predictions Vector of predicted probabilities for each class.
     * @param true_class_index Integer index of the true class.
     * @param epsilon A small constant to prevent log(0). Defaults to 1e-12.
     * @return The cross-entropy loss value.
     */
    double categorical_cross_entropy_with_index(const std::vector<double>& predictions,
                                                int true_class_index,
                                                double epsilon = 1e-12);

    /**
     * @brief Calculates the derivative of Categorical Cross-Entropy loss with respect to
     * the pre-softmax logits (z_i), assuming predictions are the output of a Softmax layer.
     * This is the common form for backpropagation: dL/dz_i = predictions[i] - targets_one_hot[i]
     * @param predictions Vector of predicted probabilities (output from Softmax).
     * @param targets_one_hot Vector of true probabilities (one-hot encoded).
     * @return Vector of derivatives dL/dz_i.
     */
    std::vector<double> categorical_cross_entropy_softmax_derivative(
        const std::vector<double>& predictions,
        const std::vector<double>& targets_one_hot);

    /**
     * @brief Calculates the derivative of Categorical Cross-Entropy loss with respect to
     * the pre-softmax logits (z_i), assuming predictions are the output of a Softmax layer,
     * and the true label is an index.
     * dL/dz_i = predictions[i] - y_i (where y_i is 1 for true_class_index, 0 otherwise)
     * @param predictions Vector of predicted probabilities (output from Softmax).
     * @param true_class_index Integer index of the true class.
     * @return Vector of derivatives dL/dz_i.
     */
    std::vector<double> categorical_cross_entropy_softmax_derivative_with_index(
        const std::vector<double>& predictions,
        int true_class_index);

} // namespace LossFunctions

#endif // LOSS_FUNCTIONS_H