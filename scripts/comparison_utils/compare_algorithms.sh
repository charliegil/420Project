#!/bin/bash

"""
Motion Estimation Algorithm Comparison Script

This script compares different motion estimation algorithms (Three Step Search,
Full Search, and Diamond Search) by processing pairs of frames and generating
a comparison video.

Usage:
    ./compare_algorithms.sh output_video.mp4 frame1.png frame2.png [block_size] [search_area]

Arguments:
    output_video.mp4: Path where the output comparison video will be saved
    frame1.png, frame2.png: Input frame images to process
    block_size: Size of blocks for motion estimation (default: 16)
    search_area: Size of search area (default: 7)
"""

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VIDEO_UTILS_DIR="$PROJECT_ROOT/scripts/video_utils"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$DATA_DIR/results"

# Create necessary directories
mkdir -p "$RESULTS_DIR"

# Check if the required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 [output_video.mp4] [frame1.png] [frame2.png] [block_size] [search_area]"
    echo "  output_video.mp4: Name of the output video file"
    echo "  frame1.png, frame2.png: Input frame images"
    echo "  block_size: integer (e.g., 4, 8, 16, 32)"
    echo "  search_area: integer (e.g., 3, 5, 7, 9, 11)"
    exit 1
fi

# Parse arguments
output_video="$RESULTS_DIR/$1"
frame1="$2"
frame2="$3"
block_size=${4:-16}
search_area=${5:-7}

# Validate input files
if [ ! -f "$frame1" ] || [ ! -f "$frame2" ]; then
    echo "Error: Input frames not found"
    exit 1
fi

# Create a temporary directory for intermediate results
temp_dir=$(mktemp -d)
echo "Created temporary directory: $temp_dir"

# Function to process frames with a specific algorithm
process_algorithm() {
    local algorithm=$1
    local flag=$2
    local output_dir="$RESULTS_DIR/${algorithm}"
    mkdir -p "$output_dir"
    
    echo "Processing frames with $algorithm..."
    
    # Run motion estimation
    "$PROJECT_ROOT/build/420Project" \
        --previous "$frame1" \
        --current "$frame2" \
        --block-size "$block_size" \
        --search-area "$search_area" \
        $flag \
        --show-images > "$temp_dir/${algorithm}_output.txt"
    
    # Move and rename output files
    mv processed_previous.png "$output_dir/${algorithm}_processed_previous.png"
    mv processed_current.png "$output_dir/${algorithm}_processed_current.png"
    mv predicted_frame.png "$output_dir/${algorithm}_predicted_frame.png"
    mv residual_frame.png "$output_dir/${algorithm}_residual_frame.png"
    mv naive_residual_frame.png "$output_dir/${algorithm}_naive_residual_frame.png"
    mv reconstructed_current_frame.png "$output_dir/${algorithm}_reconstructed_current_frame.png"
    
    # Extract metrics
    local runtime=$(grep "Runtime:" "$temp_dir/${algorithm}_output.txt" | awk '{print $2}')
    local residual=$(grep "Residual Metric:" "$temp_dir/${algorithm}_output.txt" | awk '{print $3}')
    echo "$algorithm results:"
    echo "  Runtime: $runtime seconds"
    echo "  Residual Metric: $residual"
}

# Process frames with each algorithm
process_algorithm "three_step" ""
process_algorithm "full" "--full-search"
process_algorithm "diamond" "--diamond-search"

# Create list of frames for the comparison video
frames=(
    "$frame1"
    "$frame2"
    "$RESULTS_DIR/three_step/three_step_processed_previous.png"
    "$RESULTS_DIR/three_step/three_step_processed_current.png"
    "$RESULTS_DIR/three_step/three_step_predicted_frame.png"
    "$RESULTS_DIR/three_step/three_step_residual_frame.png"
    "$RESULTS_DIR/three_step/three_step_reconstructed_current_frame.png"
    "$RESULTS_DIR/full/full_predicted_frame.png"
    "$RESULTS_DIR/full/full_residual_frame.png"
    "$RESULTS_DIR/full/full_reconstructed_current_frame.png"
    "$RESULTS_DIR/diamond/diamond_predicted_frame.png"
    "$RESULTS_DIR/diamond/diamond_residual_frame.png"
    "$RESULTS_DIR/diamond/diamond_reconstructed_current_frame.png"
)

# Create comparison video
echo "Creating comparison video..."
python3 "$VIDEO_UTILS_DIR/create_video.py" "$output_video" "${frames[@]}"

# Clean up
echo "Cleaning up temporary files..."
rm -rf "$temp_dir"

echo "Comparison complete. Results saved in $RESULTS_DIR"
echo "Comparison video saved as $output_video" 