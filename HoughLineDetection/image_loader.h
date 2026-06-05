#pragma once
#include <string>
#include <vector>

struct Image {
    int width;
    int height;
	int channels;   // Number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA)
    std::vector<unsigned char> data;    // pixel data
};

Image loadImage(const std::string& path);   
Image convertToGrayscale(const Image& image);
void saveImage(const std::string& path, const Image& image);
