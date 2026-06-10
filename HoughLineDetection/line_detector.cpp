#include "line_detector.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <cmath>

const double PI_LD = 3.14159265358979323846;

std::vector<Line> detectLines(const HoughAccumulator& accumulator, int threshold)
{
    tbb::concurrent_vector<Line> detectedLines;

    tbb::parallel_for(tbb::blocked_range<int>(1, accumulator.rhoCount - 1),
        [&](const tbb::blocked_range<int>& range)
        {
            for (int rhoIdx = range.begin(); rhoIdx < range.end(); rhoIdx++)
            {
                for (int thetaIdx = 1; thetaIdx < accumulator.thetaCount - 1; thetaIdx++)
                {
                    int votes = accumulator.at(rhoIdx, thetaIdx);

                    if (votes < threshold)
                        continue;

                    // Check 5x5 neighborhood instead of 3x3
                    bool isLocalMax = true;
                    for (int dr = -2; dr <= 2 && isLocalMax; dr++)
                    {
                        for (int dc = -2; dc <= 2 && isLocalMax; dc++)
                        {
                            if (dr == 0 && dc == 0)
                                continue;

                            int nr = rhoIdx + dr;
                            int nc = thetaIdx + dc;

                            if (nr < 0 || nr >= accumulator.rhoCount ||
                                nc < 0 || nc >= accumulator.thetaCount)
                                continue;

                            if (accumulator.at(nr, nc) >= votes)
                                isLocalMax = false;
                        }
                    }

                    if (!isLocalMax)
                        continue;

                    Line line;
                    line.rho = rhoIdx - accumulator.maxRho;
                    line.theta = thetaIdx * PI_LD / accumulator.thetaCount;
                    line.votes = votes;

                    detectedLines.push_back(line);
                }
            }
        });

    return std::vector<Line>(detectedLines.begin(), detectedLines.end());
}

Image drawLines(const Image& originalImage, const std::vector<Line>& lines)
{
    Image result = originalImage;

    for (const Line& line : lines)
    {
        double cosTheta = std::cos(line.theta);
        double sinTheta = std::sin(line.theta);

        // Draw line across entire image
        if (std::abs(sinTheta) > std::abs(cosTheta))
        {
            // More horizontal line - iterate over x
            for (int x = 0; x < result.width; x++)
            {
                int y = static_cast<int>((line.rho - x * cosTheta) / sinTheta);

                if (y < 0 || y >= result.height)
                    continue;

                int idx = (y * result.width + x) * result.channels;

                // Draw in red
                result.data[idx] = 255; // R
                result.data[idx + 1] = 0;   // G
                result.data[idx + 2] = 0;   // B
            }
        }
        else
        {
            // More vertical line - iterate over y
            for (int y = 0; y < result.height; y++)
            {
                int x = static_cast<int>((line.rho - y * sinTheta) / cosTheta);

                if (x < 0 || x >= result.width)
                    continue;

                int idx = (y * result.width + x) * result.channels;

                // Draw in red
                result.data[idx] = 255; // R
                result.data[idx + 1] = 0;   // G
                result.data[idx + 2] = 0;   // B
            }
        }
    }

    return result;
}

std::vector<Line> detectLinesSequential(const HoughAccumulator& accumulator, int threshold)
{
    std::vector<Line> detectedLines;

    for (int rhoIdx = 1; rhoIdx < accumulator.rhoCount - 1; rhoIdx++)
    {
        for (int thetaIdx = 1; thetaIdx < accumulator.thetaCount - 1; thetaIdx++)
        {
            int votes = accumulator.at(rhoIdx, thetaIdx);

            if (votes < threshold)
                continue;

            bool isLocalMax = true;
            for (int dr = -2; dr <= 2 && isLocalMax; dr++)
            {
                for (int dc = -2; dc <= 2 && isLocalMax; dc++)
                {
                    if (dr == 0 && dc == 0)
                        continue;

                    int nr = rhoIdx + dr;
                    int nc = thetaIdx + dc;

                    if (nr < 0 || nr >= accumulator.rhoCount ||
                        nc < 0 || nc >= accumulator.thetaCount)
                        continue;

                    if (accumulator.at(nr, nc) >= votes)
                        isLocalMax = false;
                }
            }

            if (!isLocalMax)
                continue;

            Line line;
            line.rho = rhoIdx - accumulator.maxRho;
            line.theta = thetaIdx * PI_LD / accumulator.thetaCount;
            line.votes = votes;

            detectedLines.push_back(line);
        }
    }

    return detectedLines;
}