#include <iostream>
#include <string>
#include <fstream>
#include "image_loader.h"
#include "edge_detector.h"
#include "hough_transform.h"
#include "line_detector.h"
#include <tbb/flow_graph.h>
#include <chrono>

// Struct to pass all data through the graph
struct PipelineData {
    Image original;
    Image gray;
    Image edges;
    HoughAccumulator accumulator;
    std::vector<Line> lines;
    std::string outputPrefix;
    int threshold;
    double timeGrayscale;
    double timeEdge;
    double timeHough;
    double timeLine;
};

void processImage(const std::string& inputPath, const std::string& outputPrefix, int threshold, std::ofstream& outFile)
{
    std::cout << "\n--- Processing: " << inputPath << " ---" << std::endl;

    Image img = loadImage(inputPath);
    std::cout << "Image loaded: " << img.width << "x" << img.height << " channels: " << img.channels << std::endl;

    tbb::flow::graph g;

    // Node 1 - Grayscale conversion
    tbb::flow::function_node<PipelineData, PipelineData> grayscaleNode(g, tbb::flow::serial,
        [](PipelineData data) {
            auto start = std::chrono::high_resolution_clock::now();
            data.gray = convertToGrayscale(data.original);
            auto end = std::chrono::high_resolution_clock::now();
            data.timeGrayscale = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << "Grayscale done: " << data.timeGrayscale << " ms" << std::endl;
            return data;
        });

    // Node 2 - Edge detection
    tbb::flow::function_node<PipelineData, PipelineData> edgeNode(g, tbb::flow::serial,
        [](PipelineData data) {
            auto start = std::chrono::high_resolution_clock::now();
            data.edges = applySobel(data.gray);
            auto end = std::chrono::high_resolution_clock::now();
            data.timeEdge = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << "Edge detection done: " << data.timeEdge << " ms" << std::endl;
            return data;
        });

    // Node 3 - Hough transform
    tbb::flow::function_node<PipelineData, PipelineData> houghNode(g, tbb::flow::serial,
        [](PipelineData data) {
            auto start = std::chrono::high_resolution_clock::now();
            data.accumulator = computeHoughTransform(data.edges);
            auto end = std::chrono::high_resolution_clock::now();
            data.timeHough = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << "Hough transform done: " << data.timeHough << " ms" << std::endl;
            return data;
        });

    // Node 4 - Line detection
    tbb::flow::function_node<PipelineData, PipelineData> lineNode(g, tbb::flow::serial,
        [](PipelineData data) {
            auto start = std::chrono::high_resolution_clock::now();
            data.lines = detectLines(data.accumulator, data.threshold);
            auto end = std::chrono::high_resolution_clock::now();
            data.timeLine = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << "Lines detected: " << data.lines.size() << " in " << data.timeLine << " ms" << std::endl;
            return data;
        });

    // Node 5 - Save results
    tbb::flow::function_node<PipelineData, tbb::flow::continue_msg> saveNode(g, tbb::flow::serial,
        [&outFile](PipelineData data) {
            Image result = drawLines(data.original, data.lines);
            saveImage(data.outputPrefix + "_grayscale.png", data.gray);
            saveImage(data.outputPrefix + "_edges.png", data.edges);
            saveImage(data.outputPrefix + "_result.png", result);

            outFile << "\n[Parallel] " << data.outputPrefix << "\n";
            outFile << "  Grayscale:  " << data.timeGrayscale << " ms\n";
            outFile << "  Edge:       " << data.timeEdge << " ms\n";
            outFile << "  Hough:      " << data.timeHough << " ms\n";
            outFile << "  Lines:      " << data.timeLine << " ms\n";

            double totalParallel = data.timeGrayscale + data.timeEdge + data.timeHough + data.timeLine;
            outFile << "  Total: " << totalParallel << " ms\n";

            std::cout << "Saved results to output folder." << std::endl;
            return tbb::flow::continue_msg();
        });

    // Connect nodes
    tbb::flow::make_edge(grayscaleNode, edgeNode);
    tbb::flow::make_edge(edgeNode, houghNode);
    tbb::flow::make_edge(houghNode, lineNode);
    tbb::flow::make_edge(lineNode, saveNode);

    PipelineData input;
    input.original = img;
    input.outputPrefix = outputPrefix;
    input.threshold = threshold;
    input.timeGrayscale = input.timeEdge = input.timeHough = input.timeLine = 0.0;

    grayscaleNode.try_put(input);
    g.wait_for_all();
}

void processImageSequential(const std::string& inputPath, int threshold, std::ofstream& outFile)
{
    std::cout << "\n--- Sequential processing: " << inputPath << " ---" << std::endl;

    Image img = loadImage(inputPath);

    // Phase 1 - Grayscale
    auto start = std::chrono::high_resolution_clock::now();
    Image gray = convertToGrayscaleSequential(img);
    auto end = std::chrono::high_resolution_clock::now();
    double timeGrayscale = std::chrono::duration<double, std::milli>(end - start).count();

    // Phase 2 - Edge detection
    start = std::chrono::high_resolution_clock::now();
    Image edges = applySobelSequential(gray);
    end = std::chrono::high_resolution_clock::now();
    double timeEdge = std::chrono::duration<double, std::milli>(end - start).count();

    // Phase 3 - Hough transform
    start = std::chrono::high_resolution_clock::now();
    HoughAccumulator accumulator = computeHoughTransformSequential(edges);
    end = std::chrono::high_resolution_clock::now();
    double timeHough = std::chrono::duration<double, std::milli>(end - start).count();

    // Phase 4 - Line detection
    start = std::chrono::high_resolution_clock::now();
    std::vector<Line> lines = detectLinesSequential(accumulator, threshold);
    end = std::chrono::high_resolution_clock::now();
    double timeLine = std::chrono::duration<double, std::milli>(end - start).count();

    double totalSequential = timeGrayscale + timeEdge + timeHough + timeLine;

    outFile << "\n[Sequential] " << inputPath << "\n";
    outFile << "  Grayscale:  " << timeGrayscale << " ms\n";
    outFile << "  Edge:       " << timeEdge << " ms\n";
    outFile << "  Hough:      " << timeHough << " ms\n";
    outFile << "  Lines:      " << timeLine << " ms\n";

    outFile << "  Total: " << totalSequential << " ms\n";
    outFile << "  Lines detected: " << lines.size() << "\n";
}

int main()
{
    std::ofstream outFile("output/results.txt");
    outFile << "=== Hough Line Detection Results ===\n";

    std::vector<std::tuple<std::string, std::string, int>> images = {
        { "images/test1.bmp", "output/test1", 190 },
        { "images/test2.png", "output/test2", 190 },
        { "images/test3.png", "output/test3", 190 },
        { "images/test4.bmp", "output/test4", 320 },
        { "images/test5.png", "output/test5", 260 }
    };

    for (const auto& img : images)
    {
        processImage(std::get<0>(img), std::get<1>(img), std::get<2>(img), outFile);
        processImageSequential(std::get<0>(img), std::get<2>(img), outFile);
    }

    outFile << "\n=== Done ===\n";
    outFile.close();

    std::cout << "Done.\n";
}