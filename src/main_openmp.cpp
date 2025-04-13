#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;

// Get luminance of image. 3 values per pixel (R, G, B) -> 1 value per pixel (brightness)
cv::Mat getLuminance(const cv::Mat& frame) {
    cv::Mat result;
    cv::cvtColor(frame, result, cv::COLOR_BGR2YCrCb);
    return result;
}

// Compute number of full blocks along horizontal and vertical axis
std::pair<int, int> getDimensions(const cv::Mat& anchor, int blockSize = 16) {
    int h = anchor.rows;
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
    auto [cx, cy] = getCenter(x, y, blockSize);
    int sx = std::max(0, cx - int(blockSize/2) - searchDimension);
    int sy = std::max(0, cy - int(blockSize/2) - searchDimension);

    cv::Rect searchArea(sx, sy,
        std::min(sx + searchDimension * 2 + blockSize, w) - sx,
        std::min(sy + searchDimension * 2 + blockSize, h) - sy);

    return previous(searchArea);
}

cv::Mat getBlockZone(const std::pair<int, int>& center, const cv::Mat& searchArea, const cv::Mat& currentBlock, int blockSize) {
    int x = center.first;
    int y = center.second;
    x = x - int(blockSize/2);
    y = y - int(blockSize/2);

    x = std::max(0, x);
    y = std::max(0, y);
    x = std::min(x, searchArea.cols - blockSize);
    y = std::min(y, searchArea.rows - blockSize);

    return searchArea(cv::Rect(x, y, blockSize, blockSize));
}

double getMAD(const cv::Mat& block1, const cv::Mat& block2) {
    cv::Mat diff;
    cv::absdiff(block1, block2, diff);
    return cv::sum(diff)[0] / (block1.rows * block2.cols);
}

cv::Mat getBestMatch(const cv::Mat& compareBlock, const cv::Mat& searchArea, int blockSize) {
    int step = 4;
    int searchCenterY = searchArea.rows / 2;
    int searchCenterX = searchArea.cols / 2;

    double minMAD = std::numeric_limits<double>::infinity();
    std::pair<int, int> minP;

    while (step >= 1) {
        vector<pair<int, int>> pointList = {
            {searchCenterX, searchCenterY},
            {searchCenterX + step, searchCenterY},
            {searchCenterX, searchCenterY + step},
            {searchCenterX + step, searchCenterY + step},
            {searchCenterX - step, searchCenterY},
            {searchCenterX, searchCenterY - step},
            {searchCenterX - step, searchCenterY - step},
            {searchCenterX + step, searchCenterY - step},
            {searchCenterX - step, searchCenterY + step}
        };

        for (const auto& p : pointList) {
            cv::Mat block = getBlockZone(p, searchArea, compareBlock, blockSize);
            double MAD = getMAD(compareBlock, block);
            if (MAD < minMAD) {
                minMAD = MAD;
                minP = p;
            }
        }

        searchCenterX = minP.first;
        searchCenterY = minP.second;
        step /= 2;
    }

    int px = std::max(0, minP.first - blockSize / 2);
    int py = std::max(0, minP.second - blockSize / 2);
    px = std::min(px, searchArea.cols - blockSize);
    py = std::min(py, searchArea.rows - blockSize);

    return searchArea(cv::Rect(px, py, blockSize, blockSize));
}

cv::Mat blockSearch(const cv::Mat& previous, const cv::Mat& current, int blockSize, int searchDimension = 7) {
    int h = previous.rows;
    int w = current.cols;
    auto [numHorizontal, numVertical] = getDimensions(previous, blockSize);

    cv::Mat predicted = cv::Mat::ones(h, w, previous.type()) * 255;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < int(numHorizontal * blockSize); y += blockSize) {
        for (int x = 0; x < int(numVertical * blockSize); x += blockSize) {
            cv::Mat currentBlock = current(cv::Rect(x, y, blockSize, blockSize));
            cv::Mat searchArea = getSearchArea(x, y, previous, blockSize, searchDimension);
            cv::Mat previousBlock = getBestMatch(currentBlock, searchArea, blockSize);

            #pragma omp critical
            previousBlock.copyTo(predicted(cv::Rect(x, y, blockSize, blockSize)));
        }
    }

    return predicted;
}

cv::Mat getResidual(const cv::Mat& target, const cv::Mat& predicted) {
    cv::Mat residual;
    cv::subtract(target, predicted, residual);
    return residual;
}

cv::Mat reconstructCurrent(const cv::Mat& residual, const cv::Mat& predicted) {
    cv::Mat reconstructed;
    cv::add(residual, predicted, reconstructed);
    return reconstructed;
}

void showImages(const vector<cv::Mat>& images) {
    string imageNames[6] = {"processedPrevious", "processedCurrent", "predictedFrame", "residualFrame", "naiveResidualFrame", "reconstructedCurrentFrame"};
    for (size_t k = 0; k < images.size(); k++) {
        cv::imshow(imageNames[k], images[k]);
    }
    cv::waitKey(0);
}

double getResidualMetric(const cv::Mat& residualFrame) {
    cv::Mat absResidual;
    cv::absdiff(residualFrame, cv::Scalar(0), absResidual);
    return cv::sum(absResidual)[0] / (residualFrame.rows * residualFrame.cols);
}

pair<cv::Mat, cv::Mat> preprocess(const cv::Mat& previous, const cv::Mat& current, int blockSize) {
    if (previous.empty() || current.empty()) {
        throw invalid_argument("Input images cannot be empty.");
    }

    cv::Mat previousLuminance = getLuminance(previous);
    cv::Mat currentLuminance = getLuminance(current);

    vector<cv::Mat> previousChannels, currentChannels;
    cv::split(previousLuminance, previousChannels);
    cv::split(currentLuminance, currentChannels);

    cv::Mat previousFrame = previousChannels[0];
    cv::Mat currentFrame = currentChannels[0];

    auto [numVertical, numHorizontal] = getDimensions(previous, blockSize);
    cv::resize(previousFrame, previousFrame, cv::Size(numHorizontal * blockSize, numVertical * blockSize));
    cv::resize(currentFrame, currentFrame, cv::Size(numHorizontal * blockSize, numVertical * blockSize));

    return make_pair(previousFrame, currentFrame);
}

int main(int argc, char* argv[]) {
    string previousPath = "./frame1.png";
    string currentPath = "./frame2.png";

    cv::Mat previous = cv::imread(previousPath);
    cv::Mat current = cv::imread(currentPath);

    if (previous.empty() || current.empty()) {
        throw runtime_error("Failed to load images");
    }

    int blockSize = 16;

    auto [processedPrevious, processedCurrent] = preprocess(previous, current, blockSize);

    auto start = chrono::high_resolution_clock::now();
    cv::Mat predictedFrame = blockSearch(processedPrevious, processedCurrent, blockSize);
    cv::Mat residualFrame = getResidual(processedCurrent, predictedFrame);
    cv::Mat naiveResidualFrame = getResidual(processedPrevious, processedCurrent);
    cv::Mat reconstructedCurrentFrame = reconstructCurrent(residualFrame, predictedFrame);
    auto end = chrono::high_resolution_clock::now();

    showImages({processedPrevious, processedCurrent, predictedFrame, residualFrame, naiveResidualFrame, reconstructedCurrentFrame});

    double residualMetric = getResidualMetric(residualFrame);
    double naiveResidualMetric = getResidualMetric(naiveResidualFrame);

    cout << "Residual Metric: " << residualMetric << endl;
    cout << "Naive Residual Metric: " << naiveResidualMetric << endl;

    chrono::duration<double> duration = end - start;
    cout << "Runtime: " << duration.count() << " seconds" << endl;

    return 0;
}