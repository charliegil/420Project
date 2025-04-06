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

# Run experiments with different block sizes
echo "=== EXPERIMENTS WITH DIFFERENT BLOCK SIZES ==="
echo ""

# Three Step Search with different block sizes
echo "--- Three Step Search Algorithm ---"
for block_size in 4 8 16 32; do
    results+=($(run_experiment "three_step" $block_size 7))
done

# Full Search with different block sizes
echo "--- Full Search Algorithm ---"
for block_size in 4 8 16 32; do
    results+=($(run_experiment "full" $block_size 7))
done

# Diamond Search with different block sizes
echo "--- Diamond Search Algorithm ---"
for block_size in 4 8 16 32; do
    results+=($(run_experiment "diamond" $block_size 7))
done

# Run experiments with different search areas
echo "=== EXPERIMENTS WITH DIFFERENT SEARCH AREAS ==="
echo ""

# Three Step Search with different search areas
echo "--- Three Step Search Algorithm ---"
for search_area in 3 5 7 9 11; do
    results+=($(run_experiment "three_step" 16 $search_area))
done

# Full Search with different search areas
echo "--- Full Search Algorithm ---"
for search_area in 3 5 7 9 11; do
    results+=($(run_experiment "full" 16 $search_area))
done

# Diamond Search with different search areas
echo "--- Diamond Search Algorithm ---"
for search_area in 3 5 7 9 11; do
    results+=($(run_experiment "diamond" 16 $search_area))
done

# Run experiments with combined parameters
echo "=== EXPERIMENTS WITH COMBINED PARAMETERS ==="
echo ""

# Three Step Search with combined parameters
echo "--- Three Step Search Algorithm ---"
for block_size in 8 32; do
    for search_area in 3 5 9; do
        results+=($(run_experiment "three_step" $block_size $search_area))
    done
done

# Full Search with combined parameters
echo "--- Full Search Algorithm ---"
for block_size in 8 32; do
    for search_area in 3 5 9; do
        results+=($(run_experiment "full" $block_size $search_area))
    done
done

# Diamond Search with combined parameters
echo "--- Diamond Search Algorithm ---"
for block_size in 8 32; do
    for search_area in 3 5 9; do
        results+=($(run_experiment "diamond" $block_size $search_area))
    done
done

echo "All experiments completed. Results saved in the results directory."
echo "Run analyze_results.sh to generate a comprehensive analysis report."

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