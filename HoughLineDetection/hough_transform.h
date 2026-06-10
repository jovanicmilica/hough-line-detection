#pragma once
#include "image_loader.h"
#include <vector>

// HoughAccumulator is a 2D accumulator for the Hough Transform, 
// where each cell counts how many edge points contribute to a particular (rho, theta) pair. 
// The size of the accumulator is determined by the number of rho and theta bins, 
// and it can be used to identify lines in the edge-detected image.
struct HoughAccumulator {
    int rhoCount;
    int thetaCount;
    double maxRho;
    std::vector<int> data;

    int& at(int rhoIdx, int thetaIdx)   
    {
        return data[rhoIdx * thetaCount + thetaIdx];
    }

    const int& at(int rhoIdx, int thetaIdx) const
    {
        return data[rhoIdx * thetaCount + thetaIdx];
    }
};

HoughAccumulator computeHoughTransform(const Image& edgeImage);     // Computes the Hough Transform accumulator from the edge-detected image

HoughAccumulator computeHoughTransformSequential(const Image& edgeImage);