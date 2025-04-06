#!/bin/bash

# Check if the required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 [algorithm] [block_size] [search_area] [--show-images]"
    echo "  algorithm: three_step, full, or diamond"
    echo "  block_size: integer (e.g., 4, 8, 16, 32)"
    echo "  search_area: integer (e.g., 3, 5, 7, 9, 11)"
    echo "  --show-images: optional flag to display images"
    exit 1
fi

# Parse arguments
algorithm=$1
block_size=$2
search_area=$3
show_images=false

# Check if show_images flag is provided
if [ $# -eq 4 ] && [ "$4" == "--show-images" ]; then
    show_images=true
fi

# Validate algorithm
if [ "$algorithm" != "three_step" ] && [ "$algorithm" != "full" ] && [ "$algorithm" != "diamond" ]; then
    echo "Error: Invalid algorithm. Must be 'three_step', 'full', or 'diamond'"
    exit 1
fi

# Validate block size
if ! [[ "$block_size" =~ ^[0-9]+$ ]]; then
    echo "Error: Block size must be an integer"
    exit 1
fi

# Validate search area
if ! [[ "$search_area" =~ ^[0-9]+$ ]]; then
    echo "Error: Search area must be an integer"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p results

# Set the search flag based on the algorithm
search_flag=""
if [ "$algorithm" == "three_step" ]; then
    search_flag=""
elif [ "$algorithm" == "full" ]; then
    search_flag="--full-search"
elif [ "$algorithm" == "diamond" ]; then
    search_flag="--diamond-search"
fi

# Set the show images flag
show_flag=""
if [ "$show_images" = true ]; then
    show_flag="--show-images"
fi

# Run the experiment
echo "Running experiment with:"
echo "  Algorithm: $algorithm"
echo "  Block size: $block_size"
echo "  Search area: $search_area"
echo "  Show images: $show_images"
echo

# Run the motion estimation program
./build/420Project $search_flag --block-size $block_size --search-area $search_area $show_flag > "results/${algorithm}_block${block_size}_search${search_area}.txt"

# Check if the experiment was successful
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully"
    echo "Results saved to: results/${algorithm}_block${block_size}_search${search_area}.txt"
else
    echo "Error: Experiment failed"
    exit 1
fi 