# Hough Line Detection 
Parallel Hough Transform for line detection in digital images using Intel TBB

## Project Description
This project implements line detection using Hough Transform with parallelization via the Intel TBB library. The project compares performance between parallel and sequential versions across multiple test images.

## Features
- Image Loading – Supports PNG, BMP, JPG formats using STB library

- Grayscale Conversion – Parallel and sequential versions

- Edge Detection – Sobel operator with parallelization

- Hough Transform – Line detection with tbb::combinable for safe parallelization

- Line Detection – Finding local maxima in the accumulator

- Line Drawing – Visualizing detected lines on the original image in red

- Vote Histogram – Visualizing vote distribution in the accumulator

- Scalability Analysis – Testing speedup with 1, 2, 4, and 6 threads

- Performance Report – Automatically generates results.txt with all measurements

## Technologies
- C++ – Core language

- Intel TBB – Parallelization (parallel_for, combinable, flow_graph, task_arena)

- STB Image – Image loading and saving (header-only)

- Chrono – Execution time measurement

## Pipeline Stages
Original Image → Grayscale → Edge Detection → Hough Transform → Line Detection → Result Image

The pipeline is implemented using TBB Flow Graph for clean stage separation.

## Parallelization Details
### Grayscale Conversion
- Partitioning: By image rows

- Each row is independent

- Speedup: 3-4x on 4 cores

### Edge Detection (Sobel)
- Partitioning: By image rows

- Each 3x3 kernel operates on independent neighborhoods

- Speedup: 3-4x on 4 cores

### Hough Transform
- Partitioning: By image rows

- Safety: tbb::combinable gives each thread a private accumulator

- Merging: All private accumulators are summed at the end

- Speedup: 3-5x on 4 cores (largest gain)

### Line Detection
- Partitioning: By accumulator cells (rho index)

- Safety: tbb::concurrent_vector allows parallel insertion

- Local maxima: 5x5 neighborhood check prevents duplicate detections

- Speedup: 2-3x (less work than previous stages)

