#include "edge_detector.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <cmath>

Image applySobel(const Image& img)
{
    Image result;
    result.width = img.width;
    result.height = img.height;
    result.channels = 1;
    result.data.resize(img.width * img.height, 0);

    // Sobel kernels
    // Gx detects horizontal changes (vertical edges)
    // Gy detects vertical changes (horizontal edges)
    const int Gx[3][3] = {
        { -1,  0,  1 },
        { -2,  0,  2 },
        { -1,  0,  1 }
    };

    const int Gy[3][3] = {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };

    tbb::parallel_for(tbb::blocked_range<int>(1, img.height - 1),
        [&](const tbb::blocked_range<int>& range)
        {
            for (int y = range.begin(); y < range.end(); y++)
            {
                for (int x = 1; x < img.width - 1; x++)
                {
                    int gx = 0;
                    int gy = 0;

                    // Apply kernels to 3x3 neighborhood
                    for (int ky = -1; ky <= 1; ky++)
                    {
                        for (int kx = -1; kx <= 1; kx++)
                        {
                            int pixel = img.data[(y + ky) * img.width + (x + kx)];
                            gx += Gx[ky + 1][kx + 1] * pixel;
                            gy += Gy[ky + 1][kx + 1] * pixel;
                        }
                    }

                    // Gradient magnitude
                    int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));

                    // Threshold - pixels above 128 are edges
                    result.data[y * img.width + x] = (magnitude > 128) ? 255 : 0;
                }
            }
        });

    return result;
}

Image applySobelSequential(const Image& img)
{
    Image result;
    result.width = img.width;
    result.height = img.height;
    result.channels = 1;
    result.data.resize(img.width * img.height, 0);

    const int Gx[3][3] = {
        { -1,  0,  1 },
        { -2,  0,  2 },
        { -1,  0,  1 }
    };

    const int Gy[3][3] = {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };

    for (int y = 1; y < img.height - 1; y++)
    {
        for (int x = 1; x < img.width - 1; x++)
        {
            int gx = 0;
            int gy = 0;

            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    int pixel = img.data[(y + ky) * img.width + (x + kx)];
                    gx += Gx[ky + 1][kx + 1] * pixel;
                    gy += Gy[ky + 1][kx + 1] * pixel;
                }
            }

            int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));
            result.data[y * img.width + x] = (magnitude > 128) ? 255 : 0;
        }
    }

    return result;
}