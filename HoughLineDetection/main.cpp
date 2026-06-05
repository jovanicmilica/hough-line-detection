#include <iostream>
#include "image_loader.h"
#include "edge_detector.h"

int main()
{
    try
    {
        // Phase 1 - Image loading and grayscale conversion
        Image img = loadImage("images/test.jpg");
        std::cout << "Image loaded: " << img.width << "x" << img.height << " channels: " << img.channels << std::endl;

        Image gray = convertToGrayscale(img);
        std::cout << "Grayscale conversion done." << std::endl;

        // Phase 2 - Edge detection
        Image edges = applySobel(gray);
        std::cout << "Edge detection done." << std::endl;

        // Save results
        saveImage("output/grayscale.png", gray);
        saveImage("output/edges.png", edges);
        std::cout << "Saved grayscale and edges to output folder." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}