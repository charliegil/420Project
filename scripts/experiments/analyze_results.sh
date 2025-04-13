#!/bin/bash

# Create results.csv with headers
echo "Algorithm,Block Size,Search Area,Residual Metric,Naive Residual Metric,Runtime (s)" > results.csv

# Function to extract metrics from a result file
extract_metrics() {
    local file=$1
    local residual=""
    local naive_residual=""
    local runtime=""
    
    if [ -f "$file" ]; then
        # Extract residual metric (case insensitive, get next line if needed)
        residual=$(grep -i "residual metric" "$file" -A 1 | tail -n 1 | tr -cd '0-9.\n')
        naive_residual=$(grep -i "naive residual metric" "$file" -A 1 | tail -n 1 | tr -cd '0-9.\n')
        runtime=$(grep -i "runtime" "$file" -A 1 | tail -n 1 | tr -cd '0-9.\n')
        
        # Validate metrics are numeric
        if [[ $residual =~ ^[0-9]+\.?[0-9]*$ ]] && 
           [[ $naive_residual =~ ^[0-9]+\.?[0-9]*$ ]] && 
           [[ $runtime =~ ^[0-9]+\.?[0-9]*$ ]]; then
            echo "$residual,$naive_residual,$runtime"
        else
            echo "invalid"
        fi
    else
        echo "invalid"
    fi
}

# Arrays to store algorithm names and their corresponding metrics
algorithms=("Three Step Search" "Diamond Search" "Full Search")
block_sizes=(4 8 16 32)
search_areas=(4 8 16 32)

# Process all result files
for result_file in results/experiment_*.txt; do
    if [ -f "$result_file" ]; then
        # Extract parameters from filename
        filename=$(basename "$result_file")
        alg=$(echo "$filename" | grep -o "alg_[^_]*" | cut -d'_' -f2)
        block=$(echo "$filename" | grep -o "block_[0-9]*" | cut -d'_' -f2)
        search=$(echo "$filename" | grep -o "search_[0-9]*" | cut -d'_' -f2)
        
        # Get metrics
        metrics=$(extract_metrics "$result_file")
        if [ "$metrics" != "invalid" ]; then
            echo "$alg,$block,$search,$metrics" >> results.csv
        fi
    fi
done

# Calculate averages and generate analysis report
echo "Motion Estimation Algorithm Analysis" > analysis_report.txt
echo "==================================" >> analysis_report.txt
echo "" >> analysis_report.txt

# Process each algorithm
for alg in "${algorithms[@]}"; do
    echo "Algorithm: $alg" >> analysis_report.txt
    echo "----------------" >> analysis_report.txt
    
    # Calculate averages
    avg_residual=$(awk -F',' -v alg="$alg" '$1==alg {sum+=$4; count++} END {if(count>0) print sum/count; else print "N/A"}' results.csv)
    avg_naive=$(awk -F',' -v alg="$alg" '$1==alg {sum+=$5; count++} END {if(count>0) print sum/count; else print "N/A"}' results.csv)
    avg_runtime=$(awk -F',' -v alg="$alg" '$1==alg {sum+=$6; count++} END {if(count>0) print sum/count; else print "N/A"}' results.csv)
    
    echo "Average Residual Metric: $avg_residual" >> analysis_report.txt
    echo "Average Naive Residual Metric: $avg_naive" >> analysis_report.txt
    echo "Average Runtime (s): $avg_runtime" >> analysis_report.txt
    echo "" >> analysis_report.txt
done

# Calculate speedups
echo "Speedup Analysis" >> analysis_report.txt
echo "---------------" >> analysis_report.txt

base_runtime=$(awk -F',' '$1=="Full Search" {sum+=$6; count++} END {if(count>0) print sum/count}' results.csv)
for alg in "Three Step Search" "Diamond Search"; do
    alg_runtime=$(awk -F',' -v alg="$alg" '$1==alg {sum+=$6; count++} END {if(count>0) print sum/count}' results.csv)
    if [ -n "$base_runtime" ] && [ -n "$alg_runtime" ] && [ "$base_runtime" != "0" ]; then
        speedup=$(echo "scale=2; $base_runtime / $alg_runtime" | bc -l)
        echo "$alg speedup vs Full Search: ${speedup}x" >> analysis_report.txt
    fi
done
echo "" >> analysis_report.txt

# Block size comparison
echo "Block Size Comparison" >> analysis_report.txt
echo "-------------------" >> analysis_report.txt

for size in "${block_sizes[@]}"; do
    echo "Block Size: $size" >> analysis_report.txt
    avg_residual=$(awk -F',' -v size="$size" '$2==size {sum+=$4; count++} END {if(count>0) print sum/count; else print "N/A"}' results.csv)
    avg_runtime=$(awk -F',' -v size="$size" '$2==size {sum+=$6; count++} END {if(count>0) print sum/count; else print "N/A"}' results.csv)
    echo "  Average Residual Metric: $avg_residual" >> analysis_report.txt
    echo "  Average Runtime (s): $avg_runtime" >> analysis_report.txt
    echo "" >> analysis_report.txt
done

echo "Analysis complete. Results saved to analysis_report.txt and results.csv" 