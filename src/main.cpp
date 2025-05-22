#include "mnist_reader.h"
#include <iostream>
#include <string>

int main() {
    try {
        
        std::string train_images_path = "../data/train-images-idx3-ubyte";
        std::string train_labels_path = "../data/train-labels-idx1-ubyte";
        std::string test_images_path = "../data/t10k-images-idx3-ubyte";
        std::string test_labels_path = "../data/t10k-labels-idx1-ubyte";

        std::cout << "Attempting to read training images..." << std::endl;
        std::vector<std::vector<unsigned char>> train_images = read_mnist_images(train_images_path);
        std::cout << "Successfully read " << train_images.size() << " training images." << std::endl;

        std::cout << "Attempting to read training labels..." << std::endl;
        std::vector<unsigned char> train_labels = read_mnist_labels(train_labels_path);
        std::cout << "Successfully read " << train_labels.size() << " training labels." << std::endl;

        // You can add similar calls for test images and labels
        // For now, let's just print a few pixels from the first image
        if (!train_images.empty()) {
            std::cout << "\nFirst 10 pixels of the first training image:\n";
            for (int i = 0; i < 10; ++i) {
                std::cout << (int)train_images[0][i] << " ";
            }
            std::cout << "\nLabel of the first training image: " << (int)train_labels[0] << std::endl;
        }


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}