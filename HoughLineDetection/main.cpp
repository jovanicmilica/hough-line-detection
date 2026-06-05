#include <iostream>
#include <string>
#include "image_loader.h"
#include "edge_detector.h"
#include "hough_transform.h"
#include "line_detector.h"

void processImage(const std::string& inputPath, const std::string& outputPrefix, int threshold)
{
    std::cout << "\n--- Processing: " << inputPath << " ---" << std::endl;

    // Phase 1 - Image loading and grayscale conversion
    Image img = loadImage(inputPath);
    std::cout << "Image loaded: " << img.width << "x" << img.height << " channels: " << img.channels << std::endl;

    Image gray = convertToGrayscale(img);
    std::cout << "Grayscale conversion done." << std::endl;

    // Phase 2 - Edge detection
    Image edges = applySobel(gray);
    std::cout << "Edge detection done." << std::endl;

    // Phase 3 - Hough transform
    HoughAccumulator accumulator = computeHoughTransform(edges);
    std::cout << "Hough transform done." << std::endl;

    // Phase 4 - Line detection
    std::vector<Line> lines = detectLines(accumulator, threshold);
    std::cout << "Lines detected: " << lines.size() << std::endl;

    // Draw lines on original image
    Image result = drawLines(img, lines);

    // Save results
    saveImage(outputPrefix + "_grayscale.png", gray);
    saveImage(outputPrefix + "_edges.png", edges);
    saveImage(outputPrefix + "_result.png", result);
    std::cout << "Saved results to output folder." << std::endl;
}

int main()
{
    int threshold = 190;
    int imageCount = 4; 

    try
    {
        for (int i = 1; i <= imageCount; i++)
        {
            std::string inputPath = "images/test" + std::to_string(i) + ".jpg";
            std::string outputPrefix = "output/test" + std::to_string(i);

            processImage(inputPath, outputPrefix, threshold);
        }

        std::cout << "\nAll images processed successfully." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}