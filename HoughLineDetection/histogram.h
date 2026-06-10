#pragma once
#include "hough_transform.h"
#include <string>

void saveHistogram(const HoughAccumulator& accumulator, const std::string& outputPath);