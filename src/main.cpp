#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <filesystem>
#include <chrono>

using namespace std;

bool debug = true;

// Function prototypes
cv::Mat getThreeStepSearchMatch(const cv::Mat& compareBlock, const cv::Mat& searchArea, int blockSize);
cv::Mat getFullSearchMatch(const cv::Mat& compareBlock, const cv::Mat& searchArea, int blockSize);
cv::Mat getDiamondSearchMatch(const cv::Mat& compareBlock, const cv::Mat& searchArea, int blockSize);

// Get luminance of image. 3 values per pixel (R, G, B) -> 1 value per pixel (brightness)
cv::Mat getLuminance(const cv::Mat& frame) {
    cv::Mat result;
    cv::cvtColor(frame, result, cv::COLOR_BGR2YCrCb);
    return result;
}

// Compute number of full blocks along horizontal and vertical axis
std::pair<int, int> getDimensions(const cv::Mat& anchor, int blockSize = 16) {
    int h = anchor.rows;image.png
    int w = anchor.cols;
    int numHorizontal = int(h / blockSize);
    int numVertical = int(w / blockSize);

    return std::make_pair(numHorizontal, numVertical);
}

// Determines center coordinate of block of pixels with x, y being coordinate of top left most pixel of block
std::pair<int, int> getCenter(int x, int y, int blockSize) {
    return std::make_pair(int(x + blockSize/2), int(y + blockSize/2));
}

// Returns search area (from previous frame) for given block in current frame
cv::Mat getSearchArea(int x, int y, const cv::Mat& previous, int blockSize, int searchDimension) {
    int h = previous.rows;
    int w = previous.cols;
    auto [cx, cy] = getCenter(x, y, blockSize);  // get center coordinate of this block

    int sx = std::max(0, cx - int(blockSize/2) - searchDimension); // ensure search area is in bounds
    int sy = std::max(0, cy - int(blockSize/2) - searchDimension);

    cv::Rect searchArea(sx, sy,
                 std::min(sx + searchDimension * 2 + blockSize, w) - sx,
                 std::min(sy + searchDimension * 2 + blockSize, h) - sy);

    return previous(searchArea);
}

// Get reference to a block inside search area for a given center position
cv::Mat getBlockZone(const std::pair<int, int>& center, const cv::Mat& searchArea, const cv::Mat& currentBlock, int blockSize) {
    int x = center.first;  // coordinates of block center
    int y = center.second;
    x = x - int(blockSize/2);  // get top left corner of block
    y = y - int(blockSize/2);

    // ensure block is within bounds
    x = std::max(0, x);
    y = std::max(0, y);
    x = std::min(x, searchArea.cols - blockSize);
    y = std::min(y, searchArea.rows - blockSize);

    // retrieve block from search area
    cv::Mat block = searchArea(cv::Rect(x, y, blockSize, blockSize));

    return block;
}

// Compute Mean Absolute Difference between 2 blocks
double getMAD(const cv::Mat& block1, const cv::Mat& block2) {
    cv::Mat diff;
    cv::absdiff(block1, block2, diff);
    return cv::sum(diff)[0] / (block1.rows * block2.cols);
}

// Return most similar block in search area of previous frame (compared to current block in current frame)
// https://en.wikipedia.org/wiki/Block-matching_algorithm
cv::Mat getBestMatch(const cv::Mat& compareBlock, const cv::Mat& searchArea, int blockSize, bool useFullSearch = false, bool useDiamondSearch = false) {
    if (useFullSearch) {
        return getFullSearchMatch(compareBlock, searchArea, blockSize);
    } else if (useDiamondSearch) {
        return getDiamondSearchMatch(compareBlock, searchArea, blockSize);
    } else {
        return getThreeStepSearchMatch(compareBlock, searchArea, blockSize);
    }
}

// Implementation of Three Step Search algorithm
cv::Mat getThreeStepSearchMatch(const cv::Mat& compareBlock, const cv::Mat& searchArea, int blockSize) {
    int step = 4;
    int searchAreaHeight = searchArea.rows;
    int searchAreaWidth = searchArea.cols;
    int searchCenterY = int(searchAreaHeight / 2);  // get center of anchor search area
    int searchCenterX = int(searchAreaWidth / 2);

    double minMAD = std::numeric_limits<double>::infinity();
    std::pair<int, int> minP;

    // For each iteration, check 9 points: Center + All neighbouring blocks (in 8 directions)
    // Start with radius = 4, and reduce until radius = 1
    while (step >= 1) {
        std::vector<std::pair<int, int>> pointList = {
                {searchCenterX, searchCenterY},  // center
                {searchCenterX + step, searchCenterY},  // right
                {searchCenterX, searchCenterY + step},  // up
                {searchCenterX + step, searchCenterY + step},  // up-right
                {searchCenterX - step, searchCenterY},  // left
                {searchCenterX, searchCenterY - step},  //  down
                {searchCenterX - step, searchCenterY - step},  // down-left
                {searchCenterX + step, searchCenterY - step},  // down-right
                {searchCenterX - step, searchCenterY + step}  // up-left
        };

        for (const auto& p : pointList) {
            cv::Mat block = getBlockZone(p, searchArea, compareBlock, blockSize);  // get block in search area
            double MAD = getMAD(compareBlock, block);  // determine MAD
            if (MAD < minMAD) {  // store point with minimum MAD
                minMAD = MAD;
                minP = p;
            }
        }

        // Use current best match as center for next iteration of search
        searchCenterX = minP.first;
        searchCenterY = minP.second;
        step = int(step/2);
    }

    // Get position of best match block
    int px = minP.first;   // center of anchor block with minimum MAD
    int py = minP.second;
    px = px - int(blockSize / 2);  // get top left corner of minP
    py = py - int(blockSize / 2);
    px = std::max(0, px);  // ensure minP is within bounds
    py = std::max(0, py);

    // Make sure we don't go out of bounds
    px = std::min(px, searchArea.cols - blockSize);
    py = std::min(py, searchArea.rows - blockSize);

    // retrieve best block from anchor search area
    cv::Mat bestMatch = searchArea(cv::Rect(px, py, blockSize, blockSize));

    return bestMatch;
}

// Implementation of Full Search algorithm
cv::Mat getFullSearchMatch(const cv::Mat& compareBlock, const cv::Mat& searchArea, int blockSize) {
    int searchAreaHeight = searchArea.rows;
    int searchAreaWidth = searchArea.cols;
    
    // Initialize variables to track minimum MAD and its position
    double minMAD = std::numeric_limits<double>::infinity();
    std::pair<int, int> minP;
    
    // Search through every possible position in the search area
    for (int y = 0; y <= searchAreaHeight - blockSize; y++) {
        for (int x = 0; x <= searchAreaWidth - blockSize; x++) {
            // Get the center point of current block
            std::pair<int, int> p = {x + blockSize/2, y + blockSize/2};
            
            // Get the block at current position
            cv::Mat block = getBlockZone(p, searchArea, compareBlock, blockSize);
            
            // Calculate MAD for current block
            double MAD = getMAD(compareBlock, block);
            
            // Update minimum if current MAD is smaller
            if (MAD < minMAD) {
                minMAD = MAD;
                minP = p;
            }
        }
    }
    
    // Get position of best match block
    int px = minP.first;   // center of anchor block with minimum MAD
    int py = minP.second;
    px = px - int(blockSize / 2);  // get top left corner of minP
    py = py - int(blockSize / 2);
    px = std::max(0, px);  // ensure minP is within bounds
    py = std::max(0, py);

    // Make sure we don't go out of bounds
    px = std::min(px, searchArea.cols - blockSize);
    py = std::min(py, searchArea.rows - blockSize);

    // retrieve best block from anchor search area
    cv::Mat bestMatch = searchArea(cv::Rect(px, py, blockSize, blockSize));

    return bestMatch;
}

// Implementation of Diamond Search algorithm
cv::Mat getDiamondSearchMatch(const cv::Mat& compareBlock, const cv::Mat& searchArea, int blockSize) {
    int searchAreaHeight = searchArea.rows;
    int searchAreaWidth = searchArea.cols;
    int searchCenterY = int(searchAreaHeight / 2);  // get center of anchor search area
    int searchCenterX = int(searchAreaWidth / 2);

    double minMAD = std::numeric_limits<double>::infinity();
    std::pair<int, int> minP;
    std::pair<int, int> currentCenter = {searchCenterX, searchCenterY};
    
    // Large Diamond Search Pattern (LDSP)
    std::vector<std::pair<int, int>> ldsp = {
        {currentCenter.first, currentCenter.second},  // center
        {currentCenter.first + 2, currentCenter.second},  // right
        {currentCenter.first, currentCenter.second + 2},  // up
        {currentCenter.first - 2, currentCenter.second},  // left
        {currentCenter.first, currentCenter.second - 2}   // down
    };
    
    // Small Diamond Search Pattern (SDSP)
    std::vector<std::pair<int, int>> sdsp = {
        {currentCenter.first, currentCenter.second},  // center
        {currentCenter.first + 1, currentCenter.second},  // right
        {currentCenter.first, currentCenter.second + 1},  // up
        {currentCenter.first - 1, currentCenter.second},  // left
        {currentCenter.first, currentCenter.second - 1}   // down
    };
    
    bool foundBetterMatch = true;
    int iterations = 0;
    const int maxIterations = 10;  // Prevent infinite loops
    
    while (foundBetterMatch && iterations < maxIterations) {
        foundBetterMatch = false;
        std::vector<std::pair<int, int>>& pattern = (iterations == 0) ? ldsp : sdsp;
        
        for (const auto& p : pattern) {
            // Ensure point is within search area bounds
            if (p.first < 0 || p.first >= searchAreaWidth || p.second < 0 || p.second >= searchAreaHeight) {
                continue;
            }
            
            cv::Mat block = getBlockZone(p, searchArea, compareBlock, blockSize);
            double MAD = getMAD(compareBlock, block);
            
            if (MAD < minMAD) {
                minMAD = MAD;
                minP = p;
                foundBetterMatch = true;
            }
        }
        
        if (foundBetterMatch) {
            currentCenter = minP;
            // Update search patterns around new center
            ldsp = {
                {currentCenter.first, currentCenter.second},
                {currentCenter.first + 2, currentCenter.second},
                {currentCenter.first, currentCenter.second + 2},
                {currentCenter.first - 2, currentCenter.second},
                {currentCenter.first, currentCenter.second - 2}
            };
            sdsp = {
                {currentCenter.first, currentCenter.second},
                {currentCenter.first + 1, currentCenter.second},
                {currentCenter.first, currentCenter.second + 1},
                {currentCenter.first - 1, currentCenter.second},
                {currentCenter.first, currentCenter.second - 1}
            };
        }
        
        iterations++;
    }
    
    // Get position of best match block
    int px = minP.first;   // center of anchor block with minimum MAD
    int py = minP.second;
    px = px - int(blockSize / 2);  // get top left corner of minP
    py = py - int(blockSize / 2);
    px = std::max(0, px);  // ensure minP is within bounds
    py = std::max(0, py);

    // Out of bounds check
    px = std::min(px, searchArea.cols - blockSize);
    py = std::min(py, searchArea.rows - blockSize);

    // retrieve best block from anchor search area
    cv::Mat bestMatch = searchArea(cv::Rect(px, py, blockSize, blockSize));

    return bestMatch;
}

// Compute predicted frame by performing block search algorithm within search area for each block in current frame
cv::Mat blockSearch(const cv::Mat& previous, const cv::Mat& current, int blockSize, int searchDimension = 7, bool useFullSearch = false, bool useDiamondSearch = false) {
    int h = previous.rows;
    int w = current.cols;
    auto [numHorizontal, numVertical] = getDimensions(previous, blockSize);

    cv::Mat predicted = cv::Mat::ones(h, w, previous.type()) * 255;  // initialize empty frame

    // For each block in current frame
    for (int y = 0; y < int(numHorizontal * blockSize); y += blockSize) {
        for (int x = 0; x < int(numVertical * blockSize); x += blockSize) {
            cv::Mat currentBlock = current(cv::Rect(x, y, blockSize, blockSize));  // get current block in current frame

            cv::Mat searchArea = getSearchArea(x, y, previous, blockSize, searchDimension);  // get search area in previous frame

            cv::Mat previousBlock = getBestMatch(currentBlock, searchArea, blockSize, useFullSearch, useDiamondSearch);  // get best block match in search area
            previousBlock.copyTo(predicted(cv::Rect(x, y, blockSize, blockSize)));  // add anchor block to predicted frame
        }
    }

    return predicted;
}

// Create residual frame by subtracting predicted frame from current frame
cv::Mat getResidual(const cv::Mat& target, const cv::Mat& predicted) {
    cv::Mat residual;
    cv::subtract(target, predicted, residual);
    return residual;
}

// Reconstruct current frame from residual frame plus predicted frame (using motion estimation)
cv::Mat reconstructCurrent(const cv::Mat& residual, const cv::Mat& predicted) {
    cv::Mat reconstructed;
    cv::add(residual, predicted, reconstructed);
    return reconstructed;
}

// Display images for debugging
void showImages(const std::vector<cv::Mat>& images) {
    string imageNames[6] = {"processedPrevious", "processedCurrent", "predictedFrame", "residualFrame", "naiveResidualFrame", "reconstructedCurrentFrame"};
    for (size_t k = 0; k < images.size(); k++) {
        cv::imshow(imageNames[k], images[k]);
    }
    cv::waitKey(0);
}

// Compute residual metric
double getResidualMetric(const cv::Mat& residualFrame) {
    cv::Mat absResidual;
    cv::absdiff(residualFrame, cv::Scalar(0), absResidual);
    return cv::sum(absResidual)[0] / (residualFrame.rows * residualFrame.cols);
}

// Get luminance of image and truncate image to match block size
std::pair<cv::Mat, cv::Mat> preprocess(const cv::Mat& previous, const cv::Mat& current, int blockSize) {
    cv::Mat previousFrame, currentFrame;

    if (previous.empty() || current.empty()) {
        throw std::invalid_argument("Input images cannot be empty.");
    }

    // Get luminance of image
    cv::Mat previousLuminance = getLuminance(previous);
    cv::Mat currentLuminance = getLuminance(current);

    // Extract channels
    std::vector<cv::Mat> previousChannels, currentChannels;
    cv::split(previousLuminance, previousChannels);
    cv::split(currentLuminance, currentChannels);

    // Conserve only brightness channel
    previousFrame  = previousChannels[0];
    currentFrame = currentChannels[0];

    // Resize frames to cleanly align with number of blocks
    auto [numVertical, numHorizontal] = getDimensions(previous, blockSize);
    cv::resize(previousFrame, previousFrame, cv::Size(int(numHorizontal * blockSize), int(numVertical * blockSize)));
    cv::resize(currentFrame, currentFrame, cv::Size(int(numHorizontal * blockSize), int(numVertical * blockSize)));

    return make_pair(previousFrame, currentFrame);
}

void motionEstimation(const cv::Mat& previousFrame, const cv::Mat& currentFrame, int blockSize, bool useFullSearch = false, bool useDiamondSearch = false, bool showImagesFlag = false, int searchAreaSize = 7) {

    // Preprocess frames (get luminance of frame and resize to make dimensions divisible by block size)
    auto [processedPrevious, processedCurrent] = preprocess(previousFrame, currentFrame, blockSize);

    cv::Mat predictedFrame = blockSearch(processedPrevious, processedCurrent, blockSize, searchAreaSize, useFullSearch, useDiamondSearch);  // compute predicted frame
    cv::Mat residualFrame = getResidual(processedCurrent, predictedFrame);  // compute residual frame
    cv::Mat naiveResidualFrame = getResidual(processedPrevious, processedCurrent);  // compute naive residual frame
    cv::Mat reconstructedCurrentFrame = reconstructCurrent(residualFrame, predictedFrame);  // reconstruct current frame

    // Save output images
    cv::imwrite("processed_previous.png", processedPrevious);
    cv::imwrite("processed_current.png", processedCurrent);
    cv::imwrite("predicted_frame.png", predictedFrame);
    cv::imwrite("residual_frame.png", residualFrame);
    cv::imwrite("naive_residual_frame.png", naiveResidualFrame);
    cv::imwrite("reconstructed_current_frame.png", reconstructedCurrentFrame);
    
    cout << "Processed previous frame saved to: processed_previous.png" << endl;
    cout << "Processed current frame saved to: processed_current.png" << endl;
    cout << "Predicted frame saved to: predicted_frame.png" << endl;
    cout << "Residual frame saved to: residual_frame.png" << endl;
    cout << "Naive residual frame saved to: naive_residual_frame.png" << endl;
    cout << "Reconstructed current frame saved to: reconstructed_current_frame.png" << endl;

    // Display images if flag
    if (showImagesFlag) {
        showImages({processedPrevious, processedCurrent, predictedFrame, residualFrame, naiveResidualFrame, reconstructedCurrentFrame});
    }

    // Compute residual metrics 
    double residualMetric = getResidualMetric(residualFrame);  // residual metric between predicted and current
    double naiveResidualMetric = getResidualMetric(naiveResidualFrame);  // residual metric between previous and current without motion estimation

    string searchMethod = useFullSearch ? "Full Search" : (useDiamondSearch ? "Diamond Search" : "Three Step Search");
    cout << "Search Method: " << searchMethod << endl;
    cout << "Residual Metric: " << to_string(residualMetric) << endl;
    cout << "Naive Residual Metric: " << to_string(naiveResidualMetric) << endl;
}

int main(int argc, char* argv[]) {
    // Default parameters
    string previousPath = "frame1.png";
    string currentPath = "frame2.png";
    int blockSize = 16;
    bool useFullSearch = false;
    bool useDiamondSearch = false;
    bool generateFramesFromVideo = false;
    bool showImagesFlag = false;
    int searchAreaSize = 7;
    string videoPath = "hands.mp4";

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--previous" && i + 1 < argc) {
            previousPath = argv[++i];
        } else if (arg == "--current" && i + 1 < argc) {
            currentPath = argv[++i];
        } else if (arg == "--block-size" && i + 1 < argc) {
            blockSize = stoi(argv[++i]);
        } else if (arg == "--search-area" && i + 1 < argc) {
            searchAreaSize = stoi(argv[++i]);
        } else if (arg == "--full-search") {
            useFullSearch = true;
        } else if (arg == "--diamond-search") {
            useDiamondSearch = true;
        } else if (arg == "--generate-frames") {
            generateFramesFromVideo = true;
        } else if (arg == "--video" && i + 1 < argc) {
            videoPath = argv[++i];
        } else if (arg == "--show-images") {
            showImagesFlag = true;
        } else if (arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  --previous <path>     Path to previous frame (default: frame1.png)" << endl;
            cout << "  --current <path>      Path to current frame (default: frame2.png)" << endl;
            cout << "  --block-size <size>   Block size in pixels (default: 16)" << endl;
            cout << "  --search-area <size>  Search area size in pixels (default: 7)" << endl;
            cout << "  --full-search         Use Full Search algorithm" << endl;
            cout << "  --diamond-search      Use Diamond Search algorithm" << endl;
            cout << "  --generate-frames     Generate frames from video" << endl;
            cout << "  --video <path>        Path to video file (default: hands.mp4)" << endl;
            cout << "  --show-images         Display the processed images (program will wait for key press)" << endl;
            cout << "  --help                Show this help message" << endl;
            return 0;
        }
    }

   
    // Load frames
    cv::Mat previous = cv::imread(previousPath);
    cv::Mat current = cv::imread(currentPath);

    if (previous.empty() || current.empty()) {
        cout << "Error: Failed to load images from:" << endl;
        cout << "  Previous: " << previousPath << endl;
        cout << "  Current: " << currentPath << endl;
        return 1;
    }

    cout << "Image dimensions:" << endl;
    cout << "  Previous: " << previous.size() << endl;
    cout << "  Current: " << current.size() << endl;
    cout << "Block size: " << blockSize << " pixels" << endl;
    cout << "Search area size: " << searchAreaSize << " pixels" << endl;
    string searchMethod = useFullSearch ? "Full Search" : (useDiamondSearch ? "Diamond Search" : "Three Step Search");
    cout << "Search algorithm: " << searchMethod << endl;

    // Run motion estimation
    auto start = chrono::high_resolution_clock::now();
    motionEstimation(previous, current, blockSize, useFullSearch, useDiamondSearch, showImagesFlag, searchAreaSize);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;
    cout << "Runtime: " << duration.count() << " seconds" << endl;

    return 0;
}