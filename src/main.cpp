#include "mnist_reader.h"
#include "data_processor.h" // 
#include <iostream>
#include <string>
#include <vector>
#include <iomanip> 

int main() {
    try {
        std::string train_images_path = "../data/train-images-idx3-ubyte/train-images-idx3-ubyte";
        std::string train_labels_path = "../data/train-labels-idx1-ubyte/train-labels-idx1-ubyte";
        std::string test_images_path = "../data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
        std::string test_labels_path = "../data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";

        // --- 1. Read Raw Data ---
        std::cout << "--- Reading Raw Data ---" << std::endl;
        std::cout << "Attempting to read training images from: " << train_images_path << std::endl;
        std::vector<std::vector<unsigned char>> raw_train_images = read_mnist_images(train_images_path);
        std::cout << "Successfully read " << raw_train_images.size() << " raw training images." << std::endl;

        std::cout << "Attempting to read training labels from: " << train_labels_path << std::endl;
        std::vector<unsigned char> raw_train_labels = read_mnist_labels(train_labels_path);
        std::cout << "Successfully read " << raw_train_labels.size() << " raw training labels." << std::endl;

        // --- 2. Process Data ---
        std::cout << "\n--- Processing Data ---" << std::endl;
        DataProcessor processor;

        std::cout << "Processing training images (normalizing to 0.0-1.0)..." << std::endl;
        std::vector<std::vector<float>> processed_train_images = processor.process_images(raw_train_images);
        if (!processed_train_images.empty()) {
             std::cout << "Successfully processed " << processed_train_images.size() << " training images." << std::endl;
             if (!processed_train_images[0].empty()){
                std::cout << "First processed image now has " << processed_train_images[0].size() << " float pixels." << std::endl;
             }
        } else {
            std::cout << "Warning: Processed training images vector is empty." << std::endl;
        }


        std::cout << "Processing training labels (converting to float)..." << std::endl;
        std::vector<float> processed_train_labels = processor.process_labels(raw_train_labels);
         if (!processed_train_labels.empty()) {
            std::cout << "Successfully processed " << processed_train_labels.size() << " training labels." << std::endl;
        } else {
            std::cout << "Warning: Processed training labels vector is empty." << std::endl;
        }
        
        std::cout << "\n--- End of Data Loading and Processing ---" << std::endl;


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}