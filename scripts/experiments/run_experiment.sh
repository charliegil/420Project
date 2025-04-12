#!/bin/bash

# Usage message
if [ $# -lt 3 ]; then
    echo "Usage: $0 [algorithm] [block_size] [search_area] [--show-images]"
    echo "  algorithm: three_step, full, or diamond"
    echo "  block_size: integer (e.g., 4, 8, 16, 32)"
    echo "  search_area: integer (e.g., 3, 5, 7, 9, 11)"
    echo "  --show-images: optional flag to display images"
    exit 1
fi

algorithm=$1
block_size=$2
search_area=$3
show_images=false

if [ $# -eq 4 ] && [ "$4" == "--show-images" ]; then
    show_images=true
fi

if [ "$algorithm" != "three_step" ] && [ "$algorithm" != "full" ] && [ "$algorithm" != "diamond" ]; then
    echo "Error: Invalid algorithm. Must be 'three_step', 'full', or 'diamond'"
    exit 1
fi

if ! [[ "$block_size" =~ ^[0-9]+$ ]]; then
    echo "Error: Block size must be an integer"
    exit 1
fi

if ! [[ "$search_area" =~ ^[0-9]+$ ]]; then
    echo "Error: Search area must be an integer"
    exit 1
fi

mkdir -p results

search_flag=""
if [ "$algorithm" == "three_step" ]; then
    search_flag=""
elif [ "$algorithm" == "full" ]; then
    search_flag="--full-search"
elif [ "$algorithm" == "diamond" ]; then
    search_flag="--diamond-search"
fi

show_flag=""
if [ "$show_images" = true ]; then
    show_flag="--show-images"
fi

echo "Running experiment with:"
echo "  Algorithm: $algorithm"
echo "  Block size: $block_size"
echo "  Search area: $search_area"
echo "  Show images: $show_images"
echo

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

"$PROJECT_ROOT/build/motion_estimation" \
    --previous "$PROJECT_ROOT/data/frames/frame1.png" \
    --current "$PROJECT_ROOT/data/frames/frame2.png" \
    --block-size "$block_size" \
    --search-area "$search_area" \
    $search_flag \
    $show_flag > "results/${algorithm}_block${block_size}_search${search_area}.txt"

if [ $? -eq 0 ]; then
    echo "Experiment completed successfully"
    echo "Results saved to: results/${algorithm}_block${block_size}_search${search_area}.txt"
else
    echo "Error: Experiment failed"
    exit 1
fi 