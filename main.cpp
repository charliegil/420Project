
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>   // for create_directories
#include <cuda_runtime.h>
#include <chrono>

namespace fs = std::filesystem;

// Convert image to luminance (Y channel)
cv::Mat getLuminance(const cv::Mat& frame) {
    cv::Mat result;
    cv::cvtColor(frame, result, cv::COLOR_BGR2YCrCb);
    return result;
}
// Compute how many blocks fit in the image
std::pair<int, int> getDimensions(const cv::Mat& anchor, int blockSize = 16) {
    int h = anchor.rows;
    int w = anchor.cols;
    int numHorizontal = h / blockSize;
    int numVertical   = w / blockSize;
    return std::make_pair(numHorizontal, numVertical);
}

// Same as the CPU implementation 
cv::Mat getResidual(const cv::Mat& target, const cv::Mat& predicted) {
    cv::Mat residual;
    cv::subtract(target, predicted, residual);
    return residual;
}

double getResidualMetric(const cv::Mat& residualFrame) {
    cv::Mat absResidual;
    cv::absdiff(residualFrame, cv::Scalar(0), absResidual);
    return cv::sum(absResidual)[0] / (residualFrame.rows * residualFrame.cols);
}

cv::Mat reconstructCurrent(const cv::Mat& residual, const cv::Mat& predicted) {
    cv::Mat reconstructed;
    cv::add(residual, predicted, reconstructed);
    return reconstructed;
}

/****************************************************
 * 2) GPU KERNEL: Three-Step Search for Each Macroblock
 ****************************************************/

/**
 * @param d_prev, d_curr    Flattened single-channel images for previous/current
 * @param d_pred            Flattened single-channel predicted output
 * @param width, height     Dimensions of the images
 * @param blockSize         Size of each macroblock
 * @param searchDimension   +/- range for searching
 * @param totalBlocks       total # macroblocks
 * @param blocksPerRow      how many blocks fit per row
 */
__global__
void kernel_blockSearch(
    const unsigned char* d_prev,
    const unsigned char* d_curr,
    unsigned char*       d_pred,
    int width, int height,
    int blockSize, int searchDimension,
    int totalBlocks, int blocksPerRow)
{
    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockId >= totalBlocks) return;

    // which macroblock row/col is this thread handling?
    int by = blockId / blocksPerRow; 
    int bx = blockId % blocksPerRow;

    // top-left corner of this block in current frame
    int x = bx * blockSize;
    int y = by * blockSize;

    // 3-step search logic:
    // We'll do something similar to your getBestMatch(...) with a step = 4,
    // searching in +/- searchDimension around the block center.
    
    // center of this block
    int cx = x + blockSize / 2;
    int cy = y + blockSize / 2;

    int step = 4;
    double minMAD = 1e30;
    int bestOffX = 0;
    int bestOffY = 0;

    while (step >= 1) {
        // we check 9 positions: center plus the 8 neighbors offset by step
        for (int dy = -step; dy <= step; dy += step) {
            for (int dx = -step; dx <= step; dx += step) {
                int candCx = cx + bestOffX + dx;
                int candCy = cy + bestOffY + dy;

                // top-left corner of candidate block
                int candX = candCx - (blockSize / 2);
                int candY = candCy - (blockSize / 2);

                // clamp within [0..(width - blockSize)] etc.
                if (candX < 0) candX = 0;
                if (candY < 0) candY = 0;
                if (candX > width  - blockSize) candX = width  - blockSize;
                if (candY > height - blockSize) candY = height - blockSize;

                // Compute MAD for [candX..candX+blockSize] vs [x..x+blockSize]
                double sumDiff = 0.0;
                for (int row = 0; row < blockSize; ++row) {
                    for (int col = 0; col < blockSize; ++col) {
                        int idxCurr = (y + row) * width + (x + col);
                        int idxPrev = (candY + row) * width + (candX + col);
                        int diff = (int)d_curr[idxCurr] - (int)d_prev[idxPrev];
                        sumDiff += abs(diff);
                    }
                }
                double madVal = sumDiff / (blockSize * blockSize);

                if (madVal < minMAD) {
                    minMAD = madVal;
                    bestOffX += dx;
                    bestOffY += dy;
                }
            }
        }
        step /= 2; 
    }

    // we have our best offset from (cx, cy)
    int finalX = (cx + bestOffX) - (blockSize / 2);
    int finalY = (cy + bestOffY) - (blockSize / 2);

    // clamp
    if (finalX < 0) finalX = 0;
    if (finalY < 0) finalY = 0;
    if (finalX > width  - blockSize) finalX = width  - blockSize;
    if (finalY > height - blockSize) finalY = height - blockSize;

    // copy that best block from d_prev -> d_pred
    for (int row = 0; row < blockSize; ++row) {
        for (int col = 0; col < blockSize; ++col) {
            int dest = (y + row) * width + (x + col);
            int src  = (finalY + row) * width + (finalX + col);
            d_pred[dest] = d_prev[src];
        }
    }
}


/****************************************************
 * 3) HOST FUNCTION: blockSearchGPU
 *    This replaces CPU blockSearch(...) 
 ****************************************************/
cv::Mat blockSearchGPU(
    const cv::Mat& previous, 
    const cv::Mat& current, 
    int blockSize,
    int searchDimension)
{
    // 1) We assume `previous` and `current` are single-channel
    //    same size, rows x cols, 8-bit (CV_8UC1).
    int width  = previous.cols;
    int height = previous.rows;

    // Flatten data
    size_t size = width * height * sizeof(unsigned char);
    std::vector<unsigned char> h_prev(width * height), h_curr(width * height);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            h_prev[r * width + c] = previous.at<uchar>(r, c);
            h_curr[r * width + c] = current.at<uchar>(r, c);
        }
    }

    // Device pointers
    unsigned char *d_prev = nullptr, *d_curr = nullptr, *d_pred = nullptr;
    cudaMalloc(&d_prev, size);
    cudaMalloc(&d_curr, size);
    cudaMalloc(&d_pred, size);

    // Copy host->device
    cudaMemcpy(d_prev, h_prev.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_curr, h_curr.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(d_pred, 255, size); // initialize predicted to 255

    // Figure out how many blocks in H/V direction
    // e.g. "numHorizontal" and "numVertical" in your CPU code
    int numHorizontal = height / blockSize; 
    int numVertical   = width  / blockSize;
    int totalBlocks   = numHorizontal * numVertical;

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (totalBlocks + threadsPerBlock - 1) / threadsPerBlock;

    kernel_blockSearch<<<blocks, threadsPerBlock>>>(
        d_prev, d_curr, d_pred,
        width, height,
        blockSize, searchDimension,
        totalBlocks, /*blocksPerRow=*/numVertical
    );
    cudaDeviceSynchronize();

    // Check errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy predicted back to host
    std::vector<unsigned char> h_pred(width * height);
    cudaMemcpy(h_pred.data(), d_pred, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_pred);

    // Convert back to cv::Mat
    cv::Mat predicted(height, width, CV_8UC1);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            predicted.at<uchar>(r, c) = h_pred[r * width + c];
        }
    }
    return predicted;
}


/****************************************************
 * 4) Preprocessing + motionEstimation 
 *  
 ****************************************************/

/**
 * Instead of  CPU-based "preprocess" and "blockSearch",
 *  call blockSearchGPU instead of blockSearch.
 */
std::pair<cv::Mat, cv::Mat> preprocessGPU(const cv::Mat& previous, const cv::Mat& current, int blockSize) {
    // same as your new code: convert to YCrCb, keep only Y channel, 
    // and resize so width/height are multiples of blockSize
    if (previous.empty() || current.empty()) {
        throw std::invalid_argument("Input images cannot be empty.");
    }

    cv::Mat prevLuminance  = getLuminance(previous);
    cv::Mat currLuminance  = getLuminance(current);

    std::vector<cv::Mat> prevCh, currCh;
    cv::split(prevLuminance, prevCh);
    cv::split(currLuminance, currCh);

    cv::Mat prevFrame = prevCh[0]; 
    cv::Mat currFrame = currCh[0]; 

    auto [numHoriz, numVert] = getDimensions(prevFrame, blockSize);
    int newH = numHoriz * blockSize;
    int newW = numVert  * blockSize;

    cv::resize(prevFrame, prevFrame, cv::Size(newW, newH));
    cv::resize(currFrame, currFrame, cv::Size(newW, newH));

    return std::make_pair(prevFrame, currFrame);
}

void motionEstimationGPU(const cv::Mat& previousFrame, const cv::Mat& currentFrame, int blockSize)
{
    // 1) Preprocess
    auto [procPrev, procCurr] = preprocessGPU(previousFrame, currentFrame, blockSize);

    // 2) GPU-based block matching
    int searchDimension = 7;
    cv::Mat predictedFrame = blockSearchGPU(procPrev, procCurr, blockSize, searchDimension);

    // 3) Compute residual, naive residual, reconstruct, metrics, etc.
    cv::Mat residualFrame      = getResidual(procCurr, predictedFrame);
    cv::Mat naiveResidualFrame = getResidual(procPrev, procCurr);
    cv::Mat reconstructFrame   = reconstructCurrent(residualFrame, predictedFrame);

    double residualMetric      = getResidualMetric(residualFrame);
    double naiveResidualMetric = getResidualMetric(naiveResidualFrame);

    std::cout << "Residual Metric (GPU): " << residualMetric << std::endl;
    std::cout << "Naive Residual Metric: " << naiveResidualMetric << std::endl;
}

int main(int argc, char** argv)
{
    // Possibly parse your input frames
    std::string prevPath = "frame1.png";
    std::string currPath = "frame2.png";
    if (argc >= 3) {
        prevPath = argv[1];
        currPath = argv[2];
    }

    cv::Mat previous = cv::imread(prevPath);
    cv::Mat current  = cv::imread(currPath);

    if (previous.empty() || current.empty()) {
        std::cerr << "Failed to load images." << std::endl;
        return -1;
    }

    int blockSize = 16;

    // TIME the GPU motionEstimation
    auto start = std::chrono::high_resolution_clock::now();
    motionEstimationGPU(previous, current, blockSize);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "GPU motion estimation time: " << elapsedMs << " ms" << std::endl;

    return 0;
}
