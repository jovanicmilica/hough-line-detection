#include <iostream>
#include "image_loader.h"

int main()
{
    try
    {
        Image img = loadImage("images/test.jpg");
        std::cout << "Image loaded: " << img.width << "x" << img.height << " channels: " << img.channels << std::endl;

        Image gray = convertToGrayscale(img);
        std::cout << "Grayscale conversion done." << std::endl;

        saveImage("output/grayscale.png", gray);
        std::cout << "Saved to output/grayscale.png" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}