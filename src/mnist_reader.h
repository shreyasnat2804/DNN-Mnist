#ifndef MNIST_READER_H
#define MNIST_READER_H

#include <string>
#include <vector> // Make sure this is included for std::vector

// Function to read MNIST images
// Returns a vector of images, where each image is a vector of unsigned chars (pixel values)
std::vector<std::vector<unsigned char>> read_mnist_images(const std::string& full_path);

// Function to read MNIST labels
// Returns a vector of unsigned chars (label values)
std::vector<unsigned char> read_mnist_labels(const std::string& full_path);

#endif // MNIST_READER_H