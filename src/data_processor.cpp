#include "data_processor.h"
#include <stdexcept> 

DataProcessor::DataProcessor() {
}

std::vector<std::vector<float>> DataProcessor::process_images(
    const std::vector<std::vector<unsigned char>>& raw_images) {
    
    std::vector<std::vector<float>> processed_images;
    processed_images.reserve(raw_images.size());

    for (const auto& raw_image : raw_images) {
        if (raw_image.empty()) {
            processed_images.push_back({}); 
            continue;
        }
        
        std::vector<float> processed_image;
        processed_image.reserve(raw_image.size());

        for (unsigned char pixel_value : raw_image) {
            // Normalize pixel value from [0, 255] to [0.0, 1.0]
            processed_image.push_back(static_cast<float>(pixel_value) / 255.0f);
        }
        processed_images.push_back(processed_image);
    }
    return processed_images;
}

std::vector<float> DataProcessor::process_labels(
    const std::vector<unsigned char>& raw_labels) {
    
    std::vector<float> processed_labels;
    processed_labels.reserve(raw_labels.size()); // Pre-allocate 

    for (unsigned char label_value : raw_labels) {
        // Convert label to float.
        // For MNIST, these are digits 0-9.
        // Further processing like one-hot encoding might be done elsewhere
        // depending on the network's expected label format.
        processed_labels.push_back(static_cast<float>(label_value));
    }
    return processed_labels;
}