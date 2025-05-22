#include "mnist_reader.h"
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm> 

uint32_t reverse_int(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

std::vector<std::vector<unsigned char>> read_mnist_images(const std::string& full_path) {
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + full_path);
    }

    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051) { // Magic number for image files
        throw std::runtime_error("Invalid MNIST image file: incorrect magic number.");
    }

    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = reverse_int(num_images);

    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    num_rows = reverse_int(num_rows);

    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    num_cols = reverse_int(num_cols);

    std::vector<std::vector<unsigned char>> images(num_images);
    uint32_t image_size = num_rows * num_cols;

    for (uint32_t i = 0; i < num_images; ++i) {
        images[i].resize(image_size);
        file.read(reinterpret_cast<char*>(images[i].data()), image_size);
        if (!file) { // Check for read errors or EOF
             throw std::runtime_error("Error reading image data from file or unexpected EOF.");
        }
    }

    file.close();
    return images;
}

std::vector<unsigned char> read_mnist_labels(const std::string& full_path) {
    std::ifstream file(full_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + full_path);
    }

    uint32_t magic_number = 0;
    uint32_t num_labels = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    if (magic_number != 2049) { // Magic number for label files
        throw std::runtime_error("Invalid MNIST label file: incorrect magic number.");
    }

    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = reverse_int(num_labels);

    std::vector<unsigned char> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);
     if (!file) { // Check for read errors or EOF
        throw std::runtime_error("Error reading label data from file or unexpected EOF.");
    }

    file.close();
    return labels;
}