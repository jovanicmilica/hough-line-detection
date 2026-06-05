#pragma once
#include "image_loader.h"

Image applySobel(const Image& grayImage);	// Applies the Sobel operator to the grayscale image and returns an edge-detected image