#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <limits>
#include <stdexcept>

using namespace std;


// Get luminance (brightness) from RGB
cv::Mat getLuminance(const cv::Mat& frame) {
    cv::Mat result;
    cv::cvtColor(frame, result, cv::COLOR_BGR2YCrCb);
    return result;
}

// Get block dimensions
std::pair<int, int> getDimensions(const cv::Mat& anchor, int blockSize = 16) {
    int h = anchor.rows;
    int w = anchor.cols;
    int numVertical = int(h / blockSize);
    int numHorizontal = int(w / blockSize);

    return std::make_pair(numVertical, numHorizontal);
}

// Get center of block
std::pair<int, int> getCenter(int x, int y, int blockSize) {
    return std::make_pair(int(x + blockSize/2), int(y + blockSize/2));
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

// CUDA kernel for calculating MAD
__device__ float calculateMAD(const unsigned char* compareBlock,
                            const unsigned char* searchArea,
                            int blockSize,
                            int searchAreaWidth,
                            int x,
                            int y,
                            int frameWidth,
                            int blockX,
                            int blockY) {
    float sum = 0.0f;
    for (int i = 0; i < blockSize; i++) {
        for (int j = 0; j < blockSize; j++) {
            int searchIdx = (y + i) * searchAreaWidth + (x + j);
            int blockIdx = (blockY + i) * frameWidth + (blockX + j);
            int diff = compareBlock[blockIdx] - searchArea[searchIdx];
            sum += abs(diff);
        }
    }
    return sum / (blockSize * blockSize);
}

// CUDA kernel for Full Search with block-level parallelism
__global__ void fullSearchKernel(const unsigned char* currentFrame,
                                const unsigned char* previousFrame,
                                int blockSize,
                                int frameHeight,
                                int frameWidth,
                                int searchDimension,
                                float* madValues,
                                int2* bestPositions) {
    // Each thread processes one block in the frame.
    int blockIdxX = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdxY = blockIdx.y * blockDim.y + threadIdx.y;

    int numBlocksX = frameWidth / blockSize;
    int numBlocksY = frameHeight / blockSize;

    // Check if this thread's block is within frame bounds.
    if (blockIdxX < numBlocksX && blockIdxY < numBlocksY) {
        int blockX = blockIdxX * blockSize;
        int blockY = blockIdxY * blockSize;

        // Define search area bounds.
        int searchStartX = max(0, blockX - searchDimension);
        int searchStartY = max(0, blockY - searchDimension);
        int searchEndX = min(frameWidth - blockSize, blockX + searchDimension);
        int searchEndY = min(frameHeight - blockSize, blockY + searchDimension);

        float minMAD = INFINITY;
        int2 bestPos = make_int2(blockX, blockY);  // Default to block's original position.

        // Search within the defined area.
        for (int searchY = searchStartY; searchY <= searchEndY; searchY++) {
            for (int searchX = searchStartX; searchX <= searchEndX; searchX++) {
                float mad = calculateMAD(currentFrame, previousFrame, blockSize,
                                      frameWidth, searchX, searchY,
                                      frameWidth, blockX, blockY);

                if (mad < minMAD) {
                    minMAD = mad;
                    bestPos = make_int2(searchX, searchY);
                }
            }
        }

        int blockIndex = blockIdxY * numBlocksX + blockIdxX;
        madValues[blockIndex] = minMAD;
        bestPositions[blockIndex] = bestPos;
    }
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

// getFullSearchMatchCUDA: Uses Full Search via CUDA to compute the best-matching blocks.
cv::Mat getFullSearchMatchCUDA(const cv::Mat& currentFrame, const cv::Mat& previousFrame,
                              int blockSize, int searchDimension) {
    // Convert to grayscale if necessary.
    cv::Mat currentGray, previousGray;
    if (currentFrame.channels() != 1) {
        cv::cvtColor(currentFrame, currentGray, cv::COLOR_BGR2GRAY);
    } else {
        currentGray = currentFrame;
    }
    if (previousFrame.channels() != 1) {
        cv::cvtColor(previousFrame, previousGray, cv::COLOR_BGR2GRAY);
    } else {
        previousGray = previousFrame;
    }

    int frameHeight = currentGray.rows;
    int frameWidth = currentGray.cols;
    int numBlocksX = frameWidth / blockSize;
    int numBlocksY = frameHeight / blockSize;
    int totalBlocks = numBlocksX * numBlocksY;

    // Allocate device memory.
    unsigned char *d_currentFrame, *d_previousFrame;
    float *d_madValues;
    int2 *d_bestPositions;

    cudaMalloc(&d_currentFrame, frameHeight * frameWidth * sizeof(unsigned char));
    cudaMalloc(&d_previousFrame, frameHeight * frameWidth * sizeof(unsigned char));
    cudaMalloc(&d_madValues, totalBlocks * sizeof(float));
    cudaMalloc(&d_bestPositions, totalBlocks * sizeof(int2));

    // Copy grayscale data to the device.
    cudaMemcpy(d_currentFrame, currentGray.data, frameHeight * frameWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_previousFrame, previousGray.data, frameHeight * frameWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Kernel launch configuration.
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((numBlocksX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (numBlocksY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel.
    fullSearchKernel<<<numBlocks, threadsPerBlock>>>(
        d_currentFrame,
        d_previousFrame,
        blockSize,
        frameHeight,
        frameWidth,
        searchDimension,
        d_madValues,
        d_bestPositions
    );

    cudaDeviceSynchronize(); // Ensure kernel execution is complete.

    // Copy results back to host.
    float* h_madValues = new float[totalBlocks];
    int2* h_bestPositions = new int2[totalBlocks];
    cudaMemcpy(h_madValues, d_madValues, totalBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bestPositions, d_bestPositions, totalBlocks * sizeof(int2), cudaMemcpyDeviceToHost);

    // Create the predicted frame (grayscale).
    cv::Mat predictedFrame = cv::Mat::zeros(frameHeight, frameWidth, currentGray.type());

    // For each block, copy the corresponding block from previousGray to predictedFrame.
    for (int i = 0; i < numBlocksY; i++) {
        for (int j = 0; j < numBlocksX; j++) {
            int blockIndex = i * numBlocksX + j;
            int2 bestPos = h_bestPositions[blockIndex];
            // Safety-check that ROI is within bounds.
            if (bestPos.x < 0 || bestPos.y < 0 || bestPos.x + blockSize > previousGray.cols || bestPos.y + blockSize > previousGray.rows)
                continue;

            cv::Rect srcRect(bestPos.x, bestPos.y, blockSize, blockSize);
            cv::Rect dstRect(j * blockSize, i * blockSize, blockSize, blockSize);
            previousGray(srcRect).copyTo(predictedFrame(dstRect));
        }
    }

    // Cleanup.
    cudaFree(d_currentFrame);
    cudaFree(d_previousFrame);
    cudaFree(d_madValues);
    cudaFree(d_bestPositions);
    delete[] h_madValues;
    delete[] h_bestPositions;

    return predictedFrame;
}
double getResidualMetric(const cv::Mat& residualFrame) {
    cv::Mat absResidual;
    cv::absdiff(residualFrame, cv::Scalar(0), absResidual);
    return cv::sum(absResidual)[0] / (residualFrame.rows * residualFrame.cols);
}
cv::Mat getResidual(const cv::Mat& target, const cv::Mat& predicted) {
    cv::Mat residual;
    cv::subtract(target, predicted, residual);
    return residual;
}

int main() {
    // Load input images.
    cv::Mat frame1 = cv::imread("frame1.png");
    cv::Mat frame2 = cv::imread("frame2.png");

    if (frame1.empty() || frame2.empty()) {
        std::cout << "Error loading images" << std::endl;
        return -1;
    }

    // Parameters.
    int blockSize = 8;
    int searchDimension = 7;

    // Convert to grayscale for consistency
    auto [gray1, gray2] = preprocess(frame1, frame2, blockSize);

    // Time the CUDA Full Search.
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat predictedFrame = getFullSearchMatchCUDA(gray2, gray1, blockSize, searchDimension);  // NOTE: current, previous
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Compute residuals.
    cv::Mat residualFrame = getResidual(gray2, predictedFrame);
    cv::Mat naiveResidualFrame = getResidual(gray1, gray2);
    // Use helper to calculate residual metrics
    double residualMAE = getResidualMetric(residualFrame);
    double naiveResidualMAE = getResidualMetric(naiveResidualFrame);

    // Save results
    cv::imwrite("output_current_frame.png", gray2);
    cv::imwrite("output_previous_frame.png", gray1);
    cv::imwrite("output_predicted_frame.png", predictedFrame);
    cv::imwrite("output_residual.png", residualFrame);
    cv::imwrite("output_naive_residual.png", naiveResidualFrame);

    // Output results
    std::cout << "Images saved as output_*.png" << std::endl;
    std::cout << "CUDA Full Search Runtime: " << duration.count() << " seconds" << std::endl;
    std::cout << "Residual Metric (Predicted vs Current): " << residualMAE << std::endl;
    std::cout << "Naive Residual Metric (Previous vs Current): " << naiveResidualMAE << std::endl;

    return 0;
}