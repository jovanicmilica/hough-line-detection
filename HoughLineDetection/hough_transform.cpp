#include "hough_transform.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <cmath>

const double PI = 3.14159265358979323846;

HoughAccumulator computeHoughTransform(const Image& edgeImage)
{
    int width = edgeImage.width;
    int height = edgeImage.height;

    // Maximum possible rho value is the diagonal of the image
    double maxRho = std::sqrt(width * width + height * height);
    int rhoCount = static_cast<int>(2 * maxRho) + 1;
    int thetaCount = 180;

    // Use combinable to avoid race conditions
    // Each thread gets its own accumulator, merged at the end
    tbb::combinable<std::vector<int>> localAccumulators([&]() {
        return std::vector<int>(rhoCount * thetaCount, 0);
        });

    tbb::parallel_for(tbb::blocked_range<int>(0, height),
        [&](const tbb::blocked_range<int>& range)
        {
            auto& localAcc = localAccumulators.local();

            for (int y = range.begin(); y < range.end(); y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Only process edge pixels
                    if (edgeImage.data[y * width + x] == 0)
                        continue;

                    // Vote for all possible lines through this pixel
                    for (int thetaIdx = 0; thetaIdx < thetaCount; thetaIdx++)
                    {
                        double theta = thetaIdx * PI / thetaCount;
                        double rho = x * std::cos(theta) + y * std::sin(theta);

                        int rhoIdx = static_cast<int>(rho + maxRho);

                        if (rhoIdx >= 0 && rhoIdx < rhoCount)
                            localAcc[rhoIdx * thetaCount + thetaIdx]++;
                    }
                }
            }
        });

    // Merge all local accumulators into final result
    HoughAccumulator accumulator;
    accumulator.rhoCount = rhoCount;
    accumulator.thetaCount = thetaCount;
    accumulator.maxRho = maxRho;
    accumulator.data.resize(rhoCount * thetaCount, 0);

    localAccumulators.combine_each([&](const std::vector<int>& localAcc)
        {
            for (int i = 0; i < rhoCount * thetaCount; i++)
                accumulator.data[i] += localAcc[i];
        });

    return accumulator;
}

HoughAccumulator computeHoughTransformSequential(const Image& edgeImage)
{
    int width = edgeImage.width;
    int height = edgeImage.height;

    double maxRho = std::sqrt(width * width + height * height);
    int rhoCount = static_cast<int>(2 * maxRho) + 1;
    int thetaCount = 180;

    HoughAccumulator accumulator;
    accumulator.rhoCount = rhoCount;
    accumulator.thetaCount = thetaCount;
    accumulator.maxRho = maxRho;
    accumulator.data.resize(rhoCount * thetaCount, 0);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (edgeImage.data[y * width + x] == 0)
                continue;

            for (int thetaIdx = 0; thetaIdx < thetaCount; thetaIdx++)
            {
                double theta = thetaIdx * PI / thetaCount;
                double rho = x * std::cos(theta) + y * std::sin(theta);

                int rhoIdx = static_cast<int>(rho + maxRho);

                if (rhoIdx >= 0 && rhoIdx < rhoCount)
                    accumulator.data[rhoIdx * thetaCount + thetaIdx]++;
            }
        }
    }

    return accumulator;
}