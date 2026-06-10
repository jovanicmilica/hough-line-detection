#include <iostream>
#include <string>
#include "image_loader.h"
#include "edge_detector.h"
#include "hough_transform.h"
#include "line_detector.h"
#include <tbb/flow_graph.h>

// Struct to pass all data through the graph
struct PipelineData {
    Image original;
    Image gray;
    Image edges;
    HoughAccumulator accumulator;
    std::vector<Line> lines;
    std::string outputPrefix;
    int threshold;
};

void processImage(const std::string& inputPath, const std::string& outputPrefix, int threshold)
{
    std::cout << "\n--- Processing: " << inputPath << " ---" << std::endl;

    // Load image before graph
    Image img = loadImage(inputPath);
    std::cout << "Image loaded: " << img.width << "x" << img.height << " channels: " << img.channels << std::endl;

    // Create graph
    tbb::flow::graph g;

    // Node 1 - Grayscale conversion
    tbb::flow::function_node<PipelineData, PipelineData> grayscaleNode(g, tbb::flow::serial,
        [](PipelineData data) {
            data.gray = convertToGrayscale(data.original);
            std::cout << "Grayscale conversion done." << std::endl;
            return data;
        });

    // Node 2 - Edge detection
    tbb::flow::function_node<PipelineData, PipelineData> edgeNode(g, tbb::flow::serial,
        [](PipelineData data) {
            data.edges = applySobel(data.gray);
            std::cout << "Edge detection done." << std::endl;
            return data;
        });

    // Node 3 - Hough transform
    tbb::flow::function_node<PipelineData, PipelineData> houghNode(g, tbb::flow::serial,
        [](PipelineData data) {
            data.accumulator = computeHoughTransform(data.edges);
            std::cout << "Hough transform done." << std::endl;
            return data;
        });

    // Node 4 - Line detection
    tbb::flow::function_node<PipelineData, PipelineData> lineNode(g, tbb::flow::serial,
        [](PipelineData data) {
            data.lines = detectLines(data.accumulator, data.threshold);
            std::cout << "Lines detected: " << data.lines.size() << std::endl;
            return data;
        });

    // Node 5 - Save results
    tbb::flow::function_node<PipelineData, tbb::flow::continue_msg> saveNode(g, tbb::flow::serial,
        [](PipelineData data) {
            Image result = drawLines(data.original, data.lines);
            saveImage(data.outputPrefix + "_grayscale.png", data.gray);
            saveImage(data.outputPrefix + "_edges.png", data.edges);
            saveImage(data.outputPrefix + "_result.png", result);
            std::cout << "Saved results to output folder." << std::endl;
            return tbb::flow::continue_msg();
        });

    // Connect nodes
    tbb::flow::make_edge(grayscaleNode, edgeNode);
    tbb::flow::make_edge(edgeNode, houghNode);
    tbb::flow::make_edge(houghNode, lineNode);
    tbb::flow::make_edge(lineNode, saveNode);

    // Start the graph
    PipelineData input;
    input.original = img;
    input.outputPrefix = outputPrefix;
    input.threshold = threshold;
    grayscaleNode.try_put(input);

    // Wait for all nodes to finish
    g.wait_for_all();
}

void processImageSequential(const std::string& inputPath, int threshold)
{
    std::cout << "\n--- Sequential processing: " << inputPath << " ---" << std::endl;

    // Phase 1 - Image loading and grayscale conversion
    Image img = loadImage(inputPath);
    Image gray = convertToGrayscaleSequential(img);
    std::cout << "Grayscale conversion done." << std::endl;

    // Phase 2 - Edge detection
    Image edges = applySobelSequential(gray);
    std::cout << "Edge detection done." << std::endl;

    // Phase 3 - Hough transform
    HoughAccumulator accumulator = computeHoughTransformSequential(edges);
    std::cout << "Hough transform done." << std::endl;

    // Phase 4 - Line detection
    std::vector<Line> lines = detectLinesSequential(accumulator, threshold);
    std::cout << "Lines detected: " << lines.size() << std::endl;
}

int main()
{
    try
    {
        std::vector<std::tuple<std::string, std::string, int>> images = {
            { "images/test1.bmp", "output/test1", 190 },
            { "images/test2.png", "output/test2", 190 },
            { "images/test3.png", "output/test3", 190 },
            { "images/test4.bmp", "output/test4", 320 },
            { "images/test5.png", "output/test5", 260 }
        };

        for (const auto& img : images)
        {
            processImage(std::get<0>(img), std::get<1>(img), std::get<2>(img));
            processImageSequential(std::get<0>(img), std::get<2>(img));
        }

        std::cout << "\nAll images processed successfully." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}