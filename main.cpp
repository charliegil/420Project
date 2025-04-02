// Code to Perform Block Matching
// Ported from Python to C++

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include <filesystem>

using namespace std;

bool debug = true;

cv::Mat YCrCb2BGR(const cv::Mat& image) {
    /**
     * Converts numpy image into from YCrCb to BGR color space
     */
    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_BGR2YCrCb);
    return result;
}

cv::Mat BGR2YCrCb(const cv::Mat& image) {
    /**
     * Converts numpy image into from BGR to YCrCb color space
     */
    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_YCrCb2BGR);
    return result;
}

std::pair<int, int> segmentImage(const cv::Mat& anchor, int blockSize = 16) {
    /**
     * Determines how many macroblocks an image is composed of
     * @param anchor: I-Frame
     * @param blockSize: Size of macroblocks in pixels
     * @return: number of rows and columns of macroblocks within
     */
    int h = anchor.rows;
    int w = anchor.cols;
    int hSegments = int(h / blockSize);
    int wSegments = int(w / blockSize);
    int totBlocks = int(hSegments * wSegments);

    return std::make_pair(hSegments, wSegments);
}

std::pair<int, int> getCenter(int x, int y, int blockSize) {
    /**
     * Determines center of a block with x, y as top left corner coordinates and blockSize as blockSize
     * @return: x, y coordinates of center of a block
     */
    return std::make_pair(int(x + blockSize/2), int(y + blockSize/2));
}

cv::Mat getAnchorSearchArea(int x, int y, const cv::Mat& anchor, int blockSize, int searchArea) {
    /**
     * Returns image of anchor search area
     * @param x, y: top left coordinate of macroblock in Current Frame
     * @param anchor: I-Frame
     * @param blockSize: size of block in pixels
     * @param searchArea: size of search area in pixels
     * @return: Image of anchor search area
     */
    int h = anchor.rows;
    int w = anchor.cols;
    auto [cx, cy] = getCenter(x, y, blockSize);

    int sx = std::max(0, cx - int(blockSize/2) - searchArea); // ensure search area is in bounds
    int sy = std::max(0, cy - int(blockSize/2) - searchArea); // and get top left corner of search area

    // slice anchor frame within bounds to produce anchor search area
    cv::Rect roi(sx, sy,
                 std::min(sx + searchArea*2 + blockSize, w) - sx,
                 std::min(sy + searchArea*2 + blockSize, h) - sy);

    return anchor(roi);
}

cv::Mat getBlockZone(const std::pair<int, int>& p, const cv::Mat& aSearch, const cv::Mat& tBlock, int blockSize) {
    /**
     * Retrieves the block searched in the anchor search area to be compared with the macroblock tBlock in the current frame
     * @param p: x,y coordinates of macroblock center from current frame
     * @param aSearch: anchor search area image
     * @param tBlock: macroblock from current frame
     * @param blockSize: size of macroblock in pixels
     * @return: macroblock from anchor
     */
    int px = p.first;  // coordinates of macroblock center
    int py = p.second;
    px = px - int(blockSize/2);  // get top left corner of macroblock
    py = py - int(blockSize/2);
    px = std::max(0, px);  // ensure macroblock is within bounds
    py = std::max(0, py);

    // Make sure we don't go out of bounds
    px = std::min(px, aSearch.cols - blockSize);
    py = std::min(py, aSearch.rows - blockSize);

    // retrieve macroblock from anchor search area
    cv::Mat aBlock = aSearch(cv::Rect(px, py, blockSize, blockSize));

    try {
        assert(aBlock.rows == tBlock.rows && aBlock.cols == tBlock.cols); // must be same shape
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "ERROR - ABLOCK SHAPE: " << aBlock.size() << " != TBLOCK SHAPE: " << tBlock.size() << std::endl;
    }

    return aBlock;
}

// Compute Mean Absolute Difference between 2 blocks
double getMAD(const cv::Mat& tBlock, const cv::Mat& aBlock) {
    cv::Mat diff;
    cv::absdiff(tBlock, aBlock, diff);
    return cv::sum(diff)[0] / (tBlock.rows * tBlock.cols);
}

// TODO replace with full search
cv::Mat getBestMatch(const cv::Mat& tBlock, const cv::Mat& aSearch, int blockSize) {
    /**
     * Implemented 3 Step Search. Read about it here: https://en.wikipedia.org/wiki/Block-matching_algorithm#Three_Step_Search
     * @param tBlock: macroblock from current frame
     * @param aSearch: anchor search area
     * @param blockSize: size of macroblock in pixels
     * @return: macroblock from anchor search area with least MAD
     */
    int step = 4;
    int ah = aSearch.rows;
    int aw = aSearch.cols;
    int acy = int(ah/2);  // get center of anchor search area
    int acx = int(aw/2);

    double minMAD = std::numeric_limits<double>::infinity();
    std::pair<int, int> minP;

    while (step >= 1) {
        std::vector<std::pair<int, int>> pointList = {
                {acx, acy},          // p1
                {acx+step, acy},     // p2
                {acx, acy+step},     // p3
                {acx+step, acy+step}, // p4
                {acx-step, acy},     // p5
                {acx, acy-step},     // p6
                {acx-step, acy-step}, // p7
                {acx+step, acy-step}, // p8
                {acx-step, acy+step}  // p9
        };  // retrieve 9 search points

        for (const auto& p : pointList) {
            cv::Mat aBlock = getBlockZone(p, aSearch, tBlock, blockSize);  // get anchor macroblock
            double MAD = getMAD(tBlock, aBlock);  // determine MAD
            if (MAD < minMAD) {  // store point with minimum MAD
                minMAD = MAD;
                minP = p;
            }
        }

        acx = minP.first;
        acy = minP.second;
        step = int(step/2);
    }

    int px = minP.first;   // center of anchor block with minimum MAD
    int py = minP.second;
    px = px - int(blockSize / 2);  // get top left corner of minP
    py = py - int(blockSize / 2);
    px = std::max(0, px);  // ensure minP is within bounds
    py = std::max(0, py);

    // Make sure we don't go out of bounds
    px = std::min(px, aSearch.cols - blockSize);
    py = std::min(py, aSearch.rows - blockSize);

    // retrieve best macroblock from anchor search area
    cv::Mat matchBlock = aSearch(cv::Rect(px, py, blockSize, blockSize));

    return matchBlock;
}

cv::Mat blockSearchBody(const cv::Mat& anchor, const cv::Mat& target, int blockSize, int searchArea = 7) {
    int h = anchor.rows;
    int w = anchor.cols;
    auto [hSegments, wSegments] = segmentImage(anchor, blockSize);

    cv::Mat predicted = cv::Mat::ones(h, w, anchor.type()) * 255;

    for (int y = 0; y < int(hSegments*blockSize); y += blockSize) {
        for (int x = 0; x < int(wSegments*blockSize); x += blockSize) {
            cv::Mat targetBlock = target(cv::Rect(x, y, blockSize, blockSize));  // get current block in current frame

            cv::Mat anchorSearchArea = getAnchorSearchArea(x, y, anchor, blockSize, searchArea);  // get search area in previous frame

            cv::Mat anchorBlock = getBestMatch(targetBlock, anchorSearchArea, blockSize);  // get best block match in search area
            anchorBlock.copyTo(predicted(cv::Rect(x, y, blockSize, blockSize)));  // add anchor block to predicted frame
        }
    }

    return predicted;
}

// Create residual frame from target frame - predicted frame
cv::Mat getResidual(const cv::Mat& target, const cv::Mat& predicted) {
    cv::Mat residual;
    cv::subtract(target, predicted, residual);
    return residual;
}

// Reconstruct target frame from residual frame plus predicted frame
cv::Mat getReconstructTarget(const cv::Mat& residual, const cv::Mat& predicted) {
    cv::Mat reconstructed;
    cv::add(residual, predicted, reconstructed);
    return reconstructed;
}

// Display images for debugging
void showImages(const std::vector<cv::Mat>& images) {
    for (size_t k = 0; k < images.size(); k++) {
        cv::imshow("Image: " + std::to_string(k), images[k]);
    }
    cv::waitKey(0);
}

// Compute residual metric
double getResidualMetric(const cv::Mat& residualFrame) {
    cv::Mat absResidual;
    cv::absdiff(residualFrame, cv::Scalar(0), absResidual);
    return cv::sum(absResidual)[0] / (residualFrame.rows * residualFrame.cols);
}

std::pair<cv::Mat, cv::Mat> preprocess(const cv::Mat& anchor, const cv::Mat& target, int blockSize) {
    cv::Mat anchorFrame, targetFrame;

    if (anchor.empty() || target.empty()) {
        throw std::invalid_argument("Input images cannot be empty");
    }

    // Get luminance (using luminance instead of color improves performance)
    cv::Mat anchorYCrCb = BGR2YCrCb(anchor);
    cv::Mat targetYCrCb = BGR2YCrCb(target);

    std::vector<cv::Mat> anchorChannels, targetChannels;
    cv::split(anchorYCrCb, anchorChannels);
    cv::split(targetYCrCb, targetChannels);

    anchorFrame = anchorChannels[0];
    targetFrame = targetChannels[0];

    // Resize frame to fit segmentation
    auto [hSegments, wSegments] = segmentImage(anchorFrame, blockSize);
    cv::resize(anchorFrame, anchorFrame, cv::Size(int(wSegments*blockSize), int(hSegments*blockSize)));
    cv::resize(targetFrame, targetFrame, cv::Size(int(wSegments*blockSize), int(hSegments*blockSize)));

    return std::make_pair(anchorFrame, targetFrame);
}

std::pair<double, cv::Mat> motionEstimation(const cv::Mat& anchorFrame, const cv::Mat& targetFrame,
                                const std::string& outfile, bool saveOutput, int blockSize) {

    // Process frames
    auto [processedAnchorFrame, processedTargetFrame] = preprocess(anchorFrame, targetFrame, blockSize);

    // Compute predicted frame, residual frame, naive residual frame and reconstruct target frame
    cv::Mat predictedFrame = blockSearchBody(processedAnchorFrame, processedTargetFrame, blockSize);
    cv::Mat residualFrame = getResidual(processedTargetFrame, predictedFrame);
    cv::Mat naiveResidualFrame = getResidual(processedAnchorFrame, processedTargetFrame);
    cv::Mat reconstructTargetFrame = getReconstructTarget(residualFrame, predictedFrame);
//    showImages({processedTargetFrame, predictedFrame, residualFrame});

    // Compute residual metrics
    double residualMetric = getResidualMetric(residualFrame);
    double naiveResidualMetric = getResidualMetric(naiveResidualFrame);

    std::string rmText = "Residual Metric: " + std::to_string(residualMetric);
    std::string nrmText = "Naive Residual Metric: " + std::to_string(naiveResidualMetric);

    // Save outputs
    if (saveOutput) {
        std::filesystem::create_directories(outfile);

        cv::imwrite(outfile + "/targetFrame.png", processedTargetFrame);
        cv::imwrite(outfile + "/predictedFrame.png", predictedFrame);
        cv::imwrite(outfile + "/residualFrame.png", residualFrame);
        cv::imwrite(outfile + "/reconstructTargetFrame.png", reconstructTargetFrame);
        cv::imwrite(outfile + "/naiveResidualFrame.png", naiveResidualFrame);

        std::ofstream resultsFile(outfile + "/results.txt");
        resultsFile << rmText << std::endl << nrmText << std::endl;
        resultsFile.close();
    }

    std::cout << rmText << std::endl;
    std::cout << nrmText << std::endl;

    return std::make_pair(residualMetric, residualFrame);
}

int main(int argc, char* argv[]) {
    /*
    // Uncomment to run generate test frames
    cv::VideoCapture cap("/Users/charliegil/CLionProjects/420Project/hands.mp4");
    if (!cap.isOpened()) {
        cout << "Error: Cannot open video." << endl;
        return 1;
    }

    cv::Mat frame1, frame2, frame3, frame4;
    cap >> frame1;
    cap >> frame2;
    cap >> frame3;
    cap >> frame4;

    if (frame1.empty() || frame2.empty()) {
        cout << "Error: Could not read frames." << endl;
        return 1;
    }

    cv::imwrite("/Users/charliegil/CLionProjects/420Project/frame1.png", frame1);
    cv::imwrite("/Users/charliegil/CLionProjects/420Project/frame2.png", frame2);
    cv::imwrite("/Users/charliegil/CLionProjects/420Project/frame3.png", frame3);
    cv::imwrite("/Users/charliegil/CLionProjects/420Project/frame4.png", frame4);
     */

    // Compute motion estimation
    std::string anchorPath = "/Users/charliegil/CLionProjects/420Project/frame1.png";
    std::string targetPath = "/Users/charliegil/CLionProjects/420Project/frame2.png";

    cv::Mat anchor = cv::imread(anchorPath);
    cv::Mat target = cv::imread(targetPath);

    if (anchor.empty() || target.empty()) {
        throw std::runtime_error("Failed to load images");
    }

    motionEstimation(anchor, target, "/Users/charliegil/CLionProjects/420Project/output", false, 8);

    return 0;
}