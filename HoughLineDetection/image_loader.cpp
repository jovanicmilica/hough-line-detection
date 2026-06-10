#include "image_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <stdexcept>

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