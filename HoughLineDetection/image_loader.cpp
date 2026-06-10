#include "image_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <stdexcept>
#include "hough_transform.h"

Image loadImage(const std::string& path)
{
    Image img;
    unsigned char* raw = stbi_load(path.c_str(), &img.width, &img.height, &img.channels, 0);

    if (!raw)
		throw std::runtime_error("Error loading image: " + path + " - " + stbi_failure_reason());

    int totalBytes = img.width * img.height * img.channels;
    img.data.assign(raw, raw + totalBytes);

    stbi_image_free(raw);
    return img;
}

Image convertToGrayscale(const Image& img)
{
    if (img.channels == 1)
        return img;

    Image gray;
    gray.width = img.width;
    gray.height = img.height;
    gray.channels = 1;
    gray.data.resize(img.width * img.height);

    tbb::parallel_for(tbb::blocked_range<int>(0, img.height),
        [&](const tbb::blocked_range<int>& range)
        {
            for (int y = range.begin(); y < range.end(); y++)
            {
                for (int x = 0; x < img.width; x++)
                {
                    int srcIdx = (y * img.width + x) * img.channels;
                    int dstIdx = y * img.width + x;

                    unsigned char r = img.data[srcIdx];
                    unsigned char g = img.data[srcIdx + 1];
                    unsigned char b = img.data[srcIdx + 2];

                    gray.data[dstIdx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
                }
            }
        });

    return gray;
}

void saveImage(const std::string& path, const Image& image)
{
    int result = stbi_write_png(path.c_str(), image.width, image.height, image.channels, image.data.data(), image.width * image.channels);

    if (!result)
		throw std::runtime_error("Error saving image: " + path);
}

Image convertToGrayscaleSequential(const Image& img)
{
    if (img.channels == 1)
        return img;

    Image gray;
    gray.width = img.width;
    gray.height = img.height;
    gray.channels = 1;
    gray.data.resize(img.width * img.height);

    for (int y = 0; y < img.height; y++)
    {
        for (int x = 0; x < img.width; x++)
        {
            int srcIdx = (y * img.width + x) * img.channels;
            int dstIdx = y * img.width + x;

            unsigned char r = img.data[srcIdx];
            unsigned char g = img.data[srcIdx + 1];
            unsigned char b = img.data[srcIdx + 2];

            gray.data[dstIdx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }

    return gray;
}

void saveHistogram(const HoughAccumulator& accumulator, const std::string& outputPath)
{
    // Count votes distribution
    int maxVotes = *std::max_element(accumulator.data.begin(), accumulator.data.end());

    int numBins = 50;
    std::vector<int> bins(numBins, 0);

    for (int val : accumulator.data)
    {
        if (val == 0) continue; // Skip empty cells
        int bin = static_cast<int>((double)val / maxVotes * (numBins - 1));
        bins[bin]++;
    }

    // Image dimensions
    int imgWidth = 600;
    int imgHeight = 400;
    int padding = 50;
    int barWidth = (imgWidth - 2 * padding) / numBins;

    std::vector<unsigned char> imgData(imgWidth * imgHeight * 3, 255); // White background

    // Find max bin count for scaling
    int maxBinCount = *std::max_element(bins.begin(), bins.end());

    // Draw bars
    for (int i = 0; i < numBins; i++)
    {
        int barHeight = static_cast<int>((double)bins[i] / maxBinCount * (imgHeight - 2 * padding));
        int x0 = padding + i * barWidth;
        int y0 = imgHeight - padding - barHeight;

        for (int y = y0; y < imgHeight - padding; y++)
        {
            for (int x = x0; x < x0 + barWidth - 1; x++)
            {
                int idx = (y * imgWidth + x) * 3;
                imgData[idx] = 70;  // R
                imgData[idx + 1] = 130; // G
                imgData[idx + 2] = 180; // B - steel blue
            }
        }
    }

    // Draw axes
    for (int x = padding; x < imgWidth - padding; x++)
    {
        int idx = ((imgHeight - padding) * imgWidth + x) * 3;
        imgData[idx] = imgData[idx + 1] = imgData[idx + 2] = 0; // Black X axis
    }
    for (int y = padding; y < imgHeight - padding; y++)
    {
        int idx = (y * imgWidth + padding) * 3;
        imgData[idx] = imgData[idx + 1] = imgData[idx + 2] = 0; // Black Y axis
    }

    stbi_write_png(outputPath.c_str(), imgWidth, imgHeight, 3, imgData.data(), imgWidth * 3);
}