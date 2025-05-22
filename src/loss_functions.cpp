#include "loss_functions.h"
#include <stdexcept>   // For std::invalid_argument, std::out_of_range
#include <numeric>     // For std::accumulate (if needed, not used here)
#include <vector>
#include <cmath>       // For std::log, std::pow, std::max, std::min

namespace LossFunctions {

    // --- Mean Squared Error (MSE) ---
    double mean_squared_error(const std::vector<double>& predictions, const std::vector<double>& targets) {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("MSE: Predictions and targets vectors must have the same size.");
        }
        if (predictions.empty()) {
            return 0.0; // No data, no error.
        }

        double sum_squared_error = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            double error = predictions[i] - targets[i];
            sum_squared_error += error * error;
        }
        return sum_squared_error / static_cast<double>(predictions.size());
    }

    std::vector<double> mean_squared_error_derivative(const std::vector<double>& predictions, const std::vector<double>& targets) {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("MSE Derivative: Predictions and targets vectors must have the same size.");
        }
        if (predictions.empty()) {
            return {};
        }

        std::vector<double> derivatives(predictions.size());
        double N = static_cast<double>(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            derivatives[i] = (2.0 / N) * (predictions[i] - targets[i]);
        }
        return derivatives;
    }

    // --- Categorical Cross-Entropy Loss ---
    double categorical_cross_entropy(const std::vector<double>& predictions,
                                     const std::vector<double>& targets_one_hot,
                                     double epsilon) {
        if (predictions.size() != targets_one_hot.size()) {
            throw std::invalid_argument("Cross-Entropy: Predictions and targets_one_hot vectors must have the same size.");
        }
        if (predictions.empty()) {
            return 0.0; // No data, no loss.
        }

        double loss = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            // Clip predictions to avoid log(0) or log(<0) and ensure they are within [epsilon, 1-epsilon]
            double pred_clipped = std::max(epsilon, predictions[i]);
            pred_clipped = std::min(1.0 - epsilon, pred_clipped); 

            if (targets_one_hot[i] > 1e-9) { // Effectively checks if target_one_hot[i] is 1 (or significantly non-zero for soft labels)
                loss += targets_one_hot[i] * std::log(pred_clipped);
            }
        }
        return -loss; // Cross-entropy is the negative sum
    }

    double categorical_cross_entropy_with_index(const std::vector<double>& predictions,
                                                int true_class_index,
                                                double epsilon) {
        if (predictions.empty()) {
            throw std::invalid_argument("Cross-Entropy (index): Predictions vector cannot be empty.");
        }
        if (true_class_index < 0 || true_class_index >= static_cast<int>(predictions.size())) {
            throw std::out_of_range("Cross-Entropy (index): true_class_index is out of bounds for predictions vector size.");
        }

        // Clip prediction to avoid log(0) or log(<0)
        double pred_clipped = std::max(epsilon, predictions[true_class_index]);
        pred_clipped = std::min(1.0 - epsilon, pred_clipped);

        return -std::log(pred_clipped);
    }

    std::vector<double> categorical_cross_entropy_softmax_derivative(
        const std::vector<double>& predictions, // These are Softmax outputs p_i
        const std::vector<double>& targets_one_hot) { // These are true labels y_i (one-hot)

        if (predictions.size() != targets_one_hot.size()) {
            throw std::invalid_argument("Cross-Entropy Derivative: Predictions and targets_one_hot vectors must have the same size.");
        }
        if (predictions.empty()) {
            return {};
        }

        std::vector<double> derivatives(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            // dL/dz_i = p_i - y_i
            derivatives[i] = predictions[i] - targets_one_hot[i];
        }
        return derivatives;
    }

    std::vector<double> categorical_cross_entropy_softmax_derivative_with_index(
        const std::vector<double>& predictions, // These are Softmax outputs p_i
        int true_class_index) { // This is the true class index k

        if (predictions.empty()) {
            throw std::invalid_argument("Cross-Entropy Derivative (index): Predictions vector cannot be empty.");
        }
        int num_classes = static_cast<int>(predictions.size());
        if (true_class_index < 0 || true_class_index >= num_classes) {
            throw std::out_of_range("Cross-Entropy Derivative (index): true_class_index is out of bounds for predictions vector size.");
        }

        std::vector<double> derivatives(num_classes);
        for (int i = 0; i < num_classes; ++i) {
            double target_val_yi = (i == true_class_index) ? 1.0 : 0.0;
            // dL/dz_i = p_i - y_i
            derivatives[i] = predictions[i] - target_val_yi;
        }
        return derivatives;
    }

} // namespace LossFunctions