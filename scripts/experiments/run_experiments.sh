#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Function to run an experiment and save results
run_experiment() {
    local algorithm=$1
    local block_size=$2
    local search_area=$3
    local output_file="results/${algorithm}_block${block_size}_search${search_area}.txt"
    
    echo "Running experiment: $algorithm, Block Size: $block_size, Search Area: $search_area"
    
    # Set the search flag based on the algorithm
    local search_flag=""
    if [ "$algorithm" == "three_step" ]; then
        search_flag=""
    elif [ "$algorithm" == "full" ]; then
        search_flag="--full-search"
    elif [ "$algorithm" == "diamond" ]; then
        search_flag="--diamond-search"
    fi
    
    # Run the experiment and save output to file
    ./build/420Project --previous frame1.png --current frame2.png --block-size $block_size --search-area $search_area $search_flag > $output_file
    
    # Extract key metrics from the output file
    local residual_metric=$(grep "Residual Metric:" $output_file | awk '{print $3}')
    local naive_residual_metric=$(grep "Naive Residual Metric:" $output_file | awk '{print $4}')
    local runtime=$(grep "Runtime:" $output_file | awk '{print $2}')
    
    # Print results to console
    echo "  Residual Metric: $residual_metric"
    echo "  Naive Residual Metric: $naive_residual_metric"
    echo "  Runtime: $runtime seconds"
    echo ""
    
    # Return the metrics for later analysis
    echo "$residual_metric $naive_residual_metric $runtime"
}

# Function to run an experiment with image display
run_experiment_with_images() {
    local algorithm=$1
    local block_size=$2
    local search_area=$3
    
    echo "Running experiment with image display: $algorithm, Block Size: $block_size, Search Area: $search_area"
    echo "Press any key in the image windows to continue..."
    
    # Set the search flag based on the algorithm
    local search_flag=""
    if [ "$algorithm" == "three_step" ]; then
        search_flag=""
    elif [ "$algorithm" == "full" ]; then
        search_flag="--full-search"
    elif [ "$algorithm" == "diamond" ]; then
        search_flag="--diamond-search"
    fi
    
    ./build/420Project --previous frame1.png --current frame2.png --block-size $block_size --search-area $search_area $search_flag --show-images
}

# Array to store results for later analysis
declare -a results

# Define block sizes (2, 4, 8, 16, 32)
block_sizes=(2 4 8 16 32)

# Define search areas (3, 14, 25, 36, 50)
search_areas=(3 14 25 36 50)

# Run experiments with different block sizes and search areas for all algorithms
echo "=== EXPERIMENTS WITH DIFFERENT BLOCK SIZES AND SEARCH AREAS ==="
echo ""

# Three Step Search
echo "--- Three Step Search Algorithm ---"
for block_size in "${block_sizes[@]}"; do
    for search_area in "${search_areas[@]}"; do
        results+=($(run_experiment "three_step" $block_size $search_area))
    done
done

# Full Search
echo "--- Full Search Algorithm ---"
for block_size in "${block_sizes[@]}"; do
    for search_area in "${search_areas[@]}"; do
        results+=($(run_experiment "full" $block_size $search_area))
    done
done

# Diamond Search
echo "--- Diamond Search Algorithm ---"
for block_size in "${block_sizes[@]}"; do
    for search_area in "${search_areas[@]}"; do
        results+=($(run_experiment "diamond" $block_size $search_area))
    done
done

# Run analyze_results.sh to generate a report
echo "=== GENERATING ANALYSIS REPORT ==="
./scripts/experiments/analyze_results.sh

echo "All experiments completed. Check the results directory for detailed output files."
echo "Analysis report has been generated."

# Handle command line arguments for showing images
if [ "$1" == "show_images" ]; then
    algorithm=$2
    block_size=$3
    search_area=$4
    
    if [ "$algorithm" == "three_step" ]; then
        run_experiment_with_images "three_step" $block_size $search_area
    elif [ "$algorithm" == "full" ]; then
        run_experiment_with_images "full" $block_size $search_area
    elif [ "$algorithm" == "diamond" ]; then
        run_experiment_with_images "diamond" $block_size $search_area
    else
        echo "Invalid algorithm. Use 'three_step', 'full', or 'diamond'."
    fi
fi 