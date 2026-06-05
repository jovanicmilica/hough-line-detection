#pragma once
#include "hough_transform.h"
#include "image_loader.h"
#include <vector>

// Line struct represents a detected line in the image, defined by its distance from the origin (rho) and its angle (theta)
// votes indicates how many edge points contributed to this line in the Hough accumulator, which can be used to filter out weak lines
struct Line {       
    double rho;
    double theta;
    int votes;
};

// goes through the Hough accumulator and identifies lines that have votes above a certain threshold, returning a list of detected lines
std::vector<Line> detectLines(const HoughAccumulator& accumulator, int threshold);  

// takes the original image and a list of detected lines, and draws these lines onto a copy of the original image, 
// returning the modified image with lines drawn
Image drawLines(const Image& originalImage, const std::vector<Line>& lines);