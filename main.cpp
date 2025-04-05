#include <opencv2/opencv.hpp>

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
cv::Mat getBestMatch(const cv::Mat& compareBlock, const cv::Mat& searchArea, int blockSize) {
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

// Compute predicted frame by performing block search algorithm within search area for each block in current frame
cv::Mat blockSearch(const cv::Mat& previous, const cv::Mat& current, int blockSize, int searchDimension = 7) {
    int h = previous.rows;
    int w = current.cols;
    auto [numHorizontal, numVertical] = getDimensions(previous, blockSize);

    cv::Mat predicted = cv::Mat::ones(h, w, previous.type()) * 255;  // initialize empty frame

    // For each block in current frame
    for (int y = 0; y < int(numHorizontal * blockSize); y += blockSize) {
        for (int x = 0; x < int(numVertical * blockSize); x += blockSize) {
            cv::Mat currentBlock = current(cv::Rect(x, y, blockSize, blockSize));  // get current block in current frame

            cv::Mat searchArea = getSearchArea(x, y, previous, blockSize, searchDimension);  // get search area in previous frame

            cv::Mat previousBlock = getBestMatch(currentBlock, searchArea, blockSize);  // get best block match in search area
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

void motionEstimation(const cv::Mat& previousFrame, const cv::Mat& currentFrame, int blockSize) {

    // Preprocess frames (get luminance of frame and resize to make dimensions divisible by block size)
    auto [processedPrevious, processedCurrent] = preprocess(previousFrame, currentFrame, blockSize);

    cv::Mat predictedFrame = blockSearch(processedPrevious, processedCurrent, blockSize);  // compute predicted frame
    cv::Mat residualFrame = getResidual(processedCurrent, predictedFrame);  // compute residual frame
    cv::Mat naiveResidualFrame = getResidual(processedPrevious, processedCurrent);  // compute naive residual frame
    cv::Mat reconstructedCurrentFrame = reconstructCurrent(residualFrame, predictedFrame);  // reconstruct current frame

    // Uncomment this line to display predicted, residual and reconstructed
        showImages({processedPrevious, processedCurrent, predictedFrame, residualFrame, naiveResidualFrame, reconstructedCurrentFrame});

    // Compute residual metrics (difference between generated frame and actual)
    double residualMetric = getResidualMetric(residualFrame);  // residual metric between predicted and current
    double naiveResidualMetric = getResidualMetric(naiveResidualFrame);  // residual metric between previous and current without motion estimation

    cout << "Residual Metric: " << to_string(residualMetric) << endl;
    cout << "Naive Residual Metric: " << to_string(naiveResidualMetric) << endl;
}

// Generate test frames from video
void generateFrames() {
    // Modify this file path to use a different video source
    cv::VideoCapture video("/Users/charliegil/CLionProjects/420Project/hands.mp4");
    if (!video.isOpened()) {
        throw runtime_error("Cannot open video.");
    }

    // Modify this to get different frames of the video
    cv::Mat frame1, frame2, frame3, frame4;
    video >> frame1;
    video >> frame2;
    video >> frame3;
    video >> frame4;

    if (frame1.empty() || frame2.empty() || frame3.empty() || frame4.empty()) {
        cout << "Error: Could not read frames." << endl;
        throw runtime_error("Could not read frames.");
    }

    // Modify these file paths to save images somewhere else
    cv::imwrite("/Users/charliegil/CLionProjects/420Project/frame1.png", frame1);
    cv::imwrite("/Users/charliegil/CLionProjects/420Project/frame2.png", frame2);
    cv::imwrite("/Users/charliegil/CLionProjects/420Project/frame3.png", frame3);
    cv::imwrite("/Users/charliegil/CLionProjects/420Project/frame4.png", frame4);
}

int main(int argc, char* argv[]) {
    // Uncomment to generate test frames from video
//    try {
//        generateFrames();
//    } catch(runtime_error& e) {
//        cout << e.what() << endl;
//        return 1;
//    }

    // Compute motion estimation
    string previousPath = "/Users/charliegil/CLionProjects/420Project/frame1.png";
    string currentPath = "/Users/charliegil/CLionProjects/420Project/frame1.png";

    cv::Mat previous = cv::imread(previousPath);
    cv::Mat current = cv::imread(currentPath);

    if (previous.empty() || current.empty()) {
        throw std::runtime_error("Failed to load images");
    }

    int blockSize = 16;

    // TODO add file paths (input frames, output) and block size to input args

    auto start = chrono::high_resolution_clock::now();
    motionEstimation(previous, current, blockSize);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;
    cout << "Runtime: " << duration.count() << " seconds" << endl;

    return 0;
}