#ifndef DATA_PROCESSOR_H
#define DATA_PROCESSOR_H

#include <vector>
#include <cstdint> // For unsigned char if needed, though vector<unsigned char> is fine

class DataProcessor {
public:
    // Constructor (if needed for any setup, otherwise default is fine)
    DataProcessor();

    // Normalizes pixel values from [0, 255] to [0.0, 1.0]
    // Input: Vector of images, where each image is a vector of unsigned char pixels
    // Output: Vector of images, where each image is a vector of float pixels
    std::vector<std::vector<float>> process_images(const std::vector<std::vector<unsigned char>>& raw_images);

    // Converts labels to a suitable float format for the network
    // (Further processing like one-hot encoding could be a separate step or class)
    // Input: Vector of unsigned char labels
    // Output: Vector of float labels
    std::vector<float> process_labels(const std::vector<unsigned char>& raw_labels);
};

#endif // DATA_PROCESSOR_H