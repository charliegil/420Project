#!/bin/bash

# Check if the required arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 [algorithm] [block_size] [search_area]"
    echo "  algorithm: three_step, full, or diamond"
    echo "  block_size: integer (e.g., 4, 8, 16, 32)"
    echo "  search_area: integer (e.g., 3, 5, 7, 9, 11)"
    exit 1
fi

# Parse arguments
algorithm=$1
block_size=$2
search_area=$3

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

# Set the search flag based on the algorithm
search_flag=""
if [ "$algorithm" == "three_step" ]; then
    search_flag="--three-step-search"
elif [ "$algorithm" == "full" ]; then
    search_flag="--full-search"
elif [ "$algorithm" == "diamond" ]; then
    search_flag="--diamond-search"
fi

# Run the motion estimation program with show-images flag
echo "Running motion estimation with:"
echo "  Algorithm: $algorithm"
echo "  Block size: $block_size"
echo "  Search area: $search_area"
echo

./build/420Project $search_flag --block-size $block_size --search-area $search_area --show-images

# Check if the program ran successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to run motion estimation"
    exit 1
fi 